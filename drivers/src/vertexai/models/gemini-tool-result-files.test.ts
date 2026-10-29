/**
 * Unit tests for attachments carried by a Gemini tool result.
 *
 * A tool can hand back an image (a chart it rendered, a screenshot, an artifact it promoted
 * into the conversation). Gemini carries those as `FunctionResponse.parts`; sending the JSON
 * response alone drops them with no error, and the model then insists it cannot see what it
 * just asked for.
 */

import type { Content } from '@google/genai';
import { type DataSource, type ExecutionOptions, PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';
import {
    conversationInlineBytes,
    GEMINI_INLINE_MEDIA_MAX_BYTES,
    GEMINI_INLINE_MEDIA_TOTAL_MAX_BYTES,
    GeminiModelDefinition,
} from './gemini.js';

const OPTIONS = { model: 'gemini-3-pro' } as unknown as ExecutionOptions;

function source(uri: string, mimeType = 'image/jpeg', bytes = [1, 2, 3]): DataSource {
    return {
        name: 'page.jpg',
        mime_type: mimeType,
        getStream: vi.fn().mockResolvedValue(
            new ReadableStream({
                start(controller) {
                    controller.enqueue(new Uint8Array(bytes));
                    controller.close();
                },
            }),
        ),
        getURL: vi.fn(),
        getURI: vi.fn().mockResolvedValue(uri),
    } as unknown as DataSource;
}

function toolSegment(files?: DataSource[]): PromptSegment {
    return {
        role: PromptRole.tool,
        tool_use_id: 'view_image',
        content: '{"artifact_path":"out/plot.png"}',
        files,
    } as unknown as PromptSegment;
}

async function createPrompt(segments: PromptSegment[]): Promise<Content[]> {
    const definition = new GeminiModelDefinition('gemini-3-pro');
    const { contents } = await definition.createPrompt(null as unknown as VertexAIDriver, segments, OPTIONS);
    return contents;
}

describe('Gemini tool result attachments', () => {
    it('passes GCS audio on a user prompt as fileData without reading or signing it', async () => {
        const file = source('gs://bucket/recording.mp3', 'audio/mpeg');
        const contents = await createPrompt([{ role: PromptRole.user, content: 'Transcribe', files: [file] }]);
        expect(contents[0].parts).toEqual([
            { text: 'Transcribe' },
            { fileData: { fileUri: 'gs://bucket/recording.mp3', mimeType: 'audio/mpeg' } },
        ]);
        expect(file.getStream).not.toHaveBeenCalled();
        expect(file.getURL).not.toHaveBeenCalled();
    });

    it('inlines an image attached to a tool result as a functionResponse part', async () => {
        const contents = await createPrompt([toolSegment([source('https://signed.example/plot.png')])]);

        expect(contents[0].parts?.[0].functionResponse).toEqual({
            name: 'view_image',
            response: { artifact_path: 'out/plot.png' },
            parts: [{ inlineData: { data: 'AQID', mimeType: 'image/jpeg' } }],
        });
    });

    it('refuses to buffer a non-GCS source past the inline budget, and stops reading it', async () => {
        const chunk = new Uint8Array(1_000_000);
        let reads = 0;
        const cancel = vi.fn();
        const file = {
            name: 'video.mp4',
            mime_type: 'video/mp4',
            getStream: vi.fn().mockResolvedValue(
                new ReadableStream<Uint8Array>({
                    pull(controller) {
                        reads++;
                        controller.enqueue(chunk);
                    },
                    cancel,
                }),
            ),
            getURL: vi.fn(),
            getURI: vi.fn().mockResolvedValue('https://bucket.s3.amazonaws.com/video.mp4?X-Amz-Signature=x'),
        } as unknown as DataSource;

        const failure = await createPrompt([{ role: PromptRole.user, content: 'Describe', files: [file] }]).catch(
            (err: unknown) => err,
        );
        expect(failure).toBeInstanceOf(Error);
        expect((failure as Error).message).toContain(
            `video.mp4 exceeds the ${GEMINI_INLINE_MEDIA_MAX_BYTES} byte limit for inline Gemini input`,
        );
        expect(cancel).toHaveBeenCalled();
        // The caller sent too much: a non-retryable 413, not an unclassified failure blamed on the provider.
        const formatted = new VertexAIDriver({ project: 'test-project', region: 'us-central1' }).formatLlumiverseError(
            failure,
            { provider: 'vertexai', model: 'gemini-3-pro', operation: 'execute' },
        );
        expect(formatted).toMatchObject({ code: 413, retryable: false, name: 'GeminiInlineMediaLimitError' });
        expect(reads).toBeLessThanOrEqual(GEMINI_INLINE_MEDIA_MAX_BYTES / chunk.byteLength + 2);
    });

    it('refuses inline media past the per-request total, even when each file is under its own limit', async () => {
        const chunk = new Uint8Array(1_000_000);
        const video = (name: string, chunks: number) => {
            let reads = 0;
            return {
                name,
                mime_type: 'video/mp4',
                getStream: vi.fn().mockResolvedValue(
                    new ReadableStream<Uint8Array>({
                        pull(controller) {
                            if (reads++ < chunks) {
                                controller.enqueue(chunk);
                            } else {
                                controller.close();
                            }
                        },
                    }),
                ),
                getURL: vi.fn(),
                getURI: vi.fn().mockResolvedValue(`https://bucket.s3.amazonaws.com/${name}`),
            } as unknown as DataSource;
        };
        const perFile = GEMINI_INLINE_MEDIA_MAX_BYTES / chunk.byteLength;
        const files = [video('a.mp4', perFile), video('b.mp4', perFile)];
        expect(2 * GEMINI_INLINE_MEDIA_MAX_BYTES).toBeGreaterThan(GEMINI_INLINE_MEDIA_TOTAL_MAX_BYTES);

        await expect(createPrompt([{ role: PromptRole.user, content: 'Compare', files }])).rejects.toThrow(
            `Inline media for this request exceeds the ${GEMINI_INLINE_MEDIA_TOTAL_MAX_BYTES} byte limit at b.mp4`,
        );
    });

    it('counts the inline media a resumed conversation already carries against the request total', async () => {
        // Four 30 MB parts sharing one string: 120 MB of history without allocating it four times.
        const base64 = 'A'.repeat(40_000_000);
        const conversation = [1, 2, 3, 4].map(() => ({
            role: 'user',
            parts: [{ inlineData: { data: base64, mimeType: 'video/mp4' } }],
        }));
        expect(conversationInlineBytes(conversation)).toBe(120_000_000);

        const chunk = new Uint8Array(1_000_000);
        let reads = 0;
        const file = {
            name: 'next.mp4',
            mime_type: 'video/mp4',
            getStream: vi.fn().mockResolvedValue(
                new ReadableStream<Uint8Array>({
                    pull(controller) {
                        if (reads++ < 40) {
                            controller.enqueue(chunk);
                        } else {
                            controller.close();
                        }
                    },
                }),
            ),
            getURL: vi.fn(),
            getURI: vi.fn().mockResolvedValue('https://bucket.s3.amazonaws.com/next.mp4'),
        } as unknown as DataSource;
        const definition = new GeminiModelDefinition('gemini-3-pro');

        await expect(
            definition.createPrompt(
                null as unknown as VertexAIDriver,
                [{ role: PromptRole.user, content: 'And this one?', files: [file] }],
                { ...OPTIONS, conversation } as ExecutionOptions,
            ),
        ).rejects.toThrow(`exceeds the ${GEMINI_INLINE_MEDIA_TOTAL_MAX_BYTES} byte limit at next.mp4`);
    });

    it('does not walk binary buffers in the history', () => {
        expect(conversationInlineBytes([{ bytes: new Uint8Array(10) }, null, 'text'])).toBe(0);
    });

    it('passes a Cloud Storage attachment by URI instead of inlining it', async () => {
        const file = source('gs://bucket/agents/run-1/out/plot.png');
        const contents = await createPrompt([toolSegment([file])]);

        expect(contents[0].parts?.[0].functionResponse?.parts).toEqual([
            { fileData: { fileUri: 'gs://bucket/agents/run-1/out/plot.png', mimeType: 'image/jpeg' } },
        ]);
        expect(file.getStream).not.toHaveBeenCalled();
    });

    it('leaves the functionResponse untouched when the tool result has no attachments', async () => {
        const contents = await createPrompt([toolSegment()]);

        expect(contents[0].parts?.[0].functionResponse).toEqual({
            name: 'view_image',
            response: { artifact_path: 'out/plot.png' },
        });
        expect(contents[0].parts?.[0].functionResponse).not.toHaveProperty('parts');
    });

    it('keeps carrying attachments on user turns', async () => {
        const segments = [
            { role: PromptRole.user, content: 'Look', files: [source('https://signed.example/page.jpg')] },
        ] as unknown as PromptSegment[];

        const contents = await createPrompt(segments);

        expect(contents[0].parts).toEqual([{ text: 'Look' }, { inlineData: { data: 'AQID', mimeType: 'image/jpeg' } }]);
    });
});
