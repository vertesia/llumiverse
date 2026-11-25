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
    getGeminiPayload,
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

describe('Gemini tool result attachments the model cannot read', () => {
    function namedSource(name: string, mimeType: string): DataSource {
        return { ...source('https://signed.example/file', mimeType), name };
    }

    async function promptFor(model: string, segments: PromptSegment[], warn = vi.fn()): Promise<Content[]> {
        const definition = new GeminiModelDefinition(model);
        const driver = { logger: { warn } } as unknown as VertexAIDriver;
        const { contents } = await definition.createPrompt(driver, segments, { model } as ExecutionOptions);
        return contents;
    }

    const note = (name: string, mimeType: string, model = 'gemini-3.5-flash') =>
        `[attachment omitted: ${name} (${mimeType}) — ${model} cannot read this type in a tool result]`;

    it('keeps readable attachments in parts and puts a note for the others in response.output', async () => {
        const audio = namedSource('report.wav', 'audio/wav');
        const archive = namedSource('logs.zip', 'Application/Zip');
        const warn = vi.fn();
        const contents = await promptFor(
            'gemini-3.5-flash',
            [
                {
                    role: PromptRole.tool,
                    tool_use_id: 'fetch',
                    content: 'Fetched',
                    files: [audio, namedSource('scan.bmp', 'image/bmp'), archive, namedSource('clip.mp4', 'video/mp4')],
                } as unknown as PromptSegment,
            ],
            warn,
        );

        expect(contents[0].parts?.[0].functionResponse).toEqual({
            name: 'fetch',
            response: { output: `Fetched\n${note('report.wav', 'audio/wav')}\n${note('logs.zip', 'application/zip')}` },
            parts: [
                { inlineData: { data: 'AQID', mimeType: 'image/bmp' } },
                { inlineData: { data: 'AQID', mimeType: 'video/mp4' } },
            ],
        });
        expect(audio.getStream).not.toHaveBeenCalled();
        expect(archive.getStream).not.toHaveBeenCalled();
        expect(warn).toHaveBeenCalledWith(
            expect.objectContaining({
                omitted_attachments: [
                    { file_name: 'report.wav', mime_type: 'audio/wav' },
                    { file_name: 'logs.zip', mime_type: 'application/zip' },
                ],
            }),
            '[llumiverse] Tool result attachments the model cannot read were replaced with a note',
        );
    });

    it('sends a JSON tool result without an output field as text in output, beside the note', async () => {
        const contents = await promptFor('gemini-3.5-flash', [toolSegment([namedSource('report.wav', 'audio/wav')])]);

        expect(contents[0].parts?.[0].functionResponse).toEqual({
            name: 'view_image',
            response: { output: `{"artifact_path":"out/plot.png"}\n${note('report.wav', 'audio/wav')}` },
        });
    });

    it('replaces every attachment with a note for Gemini 2.5, which does not read functionResponse.parts', async () => {
        const contents = await promptFor('gemini-2.5-flash', [
            {
                role: PromptRole.tool,
                tool_use_id: 'fetch',
                content: 'Fetched',
                files: [namedSource('chart.png', 'image/png')],
            } as unknown as PromptSegment,
        ]);

        expect(contents[0].parts?.[0].functionResponse).toEqual({
            name: 'fetch',
            response: { output: `Fetched\n${note('chart.png', 'image/png', 'gemini-2.5-flash')}` },
        });
    });

    it('replaces an unreadable attachment stored in a replayed history before the request goes out', () => {
        // A tool result stored by an earlier turn (or by a release without this filter) is replayed every turn.
        const history: Content[] = [
            { role: 'user', parts: [{ text: 'Fetch the meeting recording' }] },
            { role: 'model', parts: [{ functionCall: { name: 'fetch', args: {} } }] },
            {
                role: 'user',
                parts: [
                    {
                        functionResponse: {
                            name: 'fetch',
                            response: { output: 'Fetched' },
                            parts: [
                                { inlineData: { data: 'AQID', mimeType: 'audio/mpeg', displayName: 'memo.mp3' } },
                                { inlineData: { data: 'AQID', mimeType: 'image/png' } },
                            ],
                        },
                    },
                ],
            },
            { role: 'model', parts: [{ text: 'It is a recording.' }] },
            { role: 'user', parts: [{ text: 'Summarise it' }] },
        ];
        const warn = vi.fn();
        const options = {
            model: 'gemini-3.5-flash',
            tools: [{ name: 'fetch', description: '', input_schema: { type: 'object', properties: {} } }],
        } as unknown as ExecutionOptions;

        const payload = getGeminiPayload(options, { contents: history }, 'execute', {
            warn,
        } as unknown as Parameters<typeof getGeminiPayload>[3]);

        const contents = payload.contents as Content[];
        expect(contents[2].parts?.[0].functionResponse).toEqual({
            name: 'fetch',
            response: { output: `Fetched\n${note('memo.mp3', 'audio/mpeg')}` },
            parts: [{ inlineData: { data: 'AQID', mimeType: 'image/png' } }],
        });
        // The caller's history is not modified.
        expect(history[2].parts?.[0].functionResponse?.parts).toHaveLength(2);
        expect(warn).toHaveBeenCalledWith(
            { model: 'gemini-3.5-flash', omitted_attachments: [{ file_name: 'memo.mp3', mime_type: 'audio/mpeg' }] },
            '[VertexAI] Stored Gemini tool result attachments the model cannot read were replaced with a note',
        );
    });
});
