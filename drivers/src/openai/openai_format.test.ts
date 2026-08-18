/**
 * Unit tests for the Responses API prompt formatter, focused on attachments carried by a
 * tool result.
 *
 * A tool can hand back an image (a chart it rendered, a screenshot, an artifact it promoted
 * into the conversation). The Responses API represents that as a content list on
 * `function_call_output`; emitting only the text drops the image with no error, and the model
 * then insists it cannot see what it just asked for.
 */

import { type DataSource, type PromptOptions, PromptRole, type PromptSegment } from '@llumiverse/common';
import { describe, expect, it, vi } from 'vitest';
import { formatOpenAILikeMultimodalPrompt } from './openai_format.js';

const OPTIONS = { model: 'gpt-5.6' } as unknown as PromptOptions;

function source(name: string, mimeType: string | undefined, bytes = [1, 2, 3]): DataSource {
    return {
        name,
        mime_type: mimeType as string,
        getStream: vi.fn().mockResolvedValue(
            new ReadableStream({
                start(controller) {
                    controller.enqueue(new Uint8Array(bytes));
                    controller.close();
                },
            }),
        ),
        getURL: vi.fn(),
        getURI: vi.fn(),
    } as unknown as DataSource;
}

function toolSegment(files?: DataSource[], content = '{"artifact_path":"out/plot.png"}'): PromptSegment {
    return { role: PromptRole.tool, tool_use_id: 'call_1', content, files } as unknown as PromptSegment;
}

function functionCallOutput(prompt: Awaited<ReturnType<typeof formatOpenAILikeMultimodalPrompt>>) {
    const item = prompt.find((i) => 'type' in i && i.type === 'function_call_output');
    if (!item) throw new Error('no function_call_output in prompt');
    return item as { call_id: string; output: string | Array<Record<string, unknown>> };
}

describe('formatOpenAILikeMultimodalPrompt - tool result attachments', () => {
    it('emits an image attached to a tool result as an input_image content part', async () => {
        const prompt = await formatOpenAILikeMultimodalPrompt(
            [toolSegment([source('plot.png', 'image/png')])],
            OPTIONS,
        );

        const output = functionCallOutput(prompt).output;
        expect(Array.isArray(output)).toBe(true);
        expect(output).toEqual([
            { type: 'input_text', text: '{"artifact_path":"out/plot.png"}' },
            { type: 'input_image', image_url: 'data:image/png;base64,AQID', detail: 'auto' },
        ]);
    });

    it('keeps the tool text first so the result reads before its attachments', async () => {
        const prompt = await formatOpenAILikeMultimodalPrompt(
            [toolSegment([source('a.png', 'image/png'), source('b.png', 'image/png')])],
            OPTIONS,
        );

        const output = functionCallOutput(prompt).output as Array<Record<string, unknown>>;
        expect(output.map((part) => part.type)).toEqual(['input_text', 'input_image', 'input_image']);
    });

    it('keeps the plain-string output when the tool result has no attachments', async () => {
        const prompt = await formatOpenAILikeMultimodalPrompt([toolSegment()], OPTIONS);

        expect(functionCallOutput(prompt).output).toBe('{"artifact_path":"out/plot.png"}');
    });

    it('omits the text part when a tool returns an attachment and no text', async () => {
        const prompt = await formatOpenAILikeMultimodalPrompt(
            [toolSegment([source('plot.png', 'image/png')], '')],
            OPTIONS,
        );

        expect(functionCallOutput(prompt).output).toEqual([
            { type: 'input_image', image_url: 'data:image/png;base64,AQID', detail: 'auto' },
        ]);
    });

    it('skips a media type the API cannot carry instead of mislabelling it as an image', async () => {
        // The tool-result attachment resolver lets audio/video through; sending either as
        // `input_image` is rejected by the API, which would fail the whole turn.
        const prompt = await formatOpenAILikeMultimodalPrompt(
            [toolSegment([source('clip.mp4', 'video/mp4')])],
            OPTIONS,
        );

        expect(functionCallOutput(prompt).output).toBe('{"artifact_path":"out/plot.png"}');
    });

    it('still requires a tool_use_id', async () => {
        const segments = [{ role: PromptRole.tool, content: 'result' }] as unknown as PromptSegment[];

        await expect(formatOpenAILikeMultimodalPrompt(segments, OPTIONS)).rejects.toThrow(/tool use id/i);
    });
});

describe('formatOpenAILikeMultimodalPrompt - attachment types', () => {
    it('maps a PDF to input_file rather than input_image', async () => {
        const segments = [
            { role: PromptRole.user, content: 'Read this', files: [source('report.pdf', 'application/pdf')] },
        ] as unknown as PromptSegment[];

        const prompt = await formatOpenAILikeMultimodalPrompt(segments, OPTIONS);

        expect(prompt[0]).toMatchObject({
            content: [
                {
                    type: 'input_file',
                    filename: 'report.pdf',
                    file_data: 'data:application/pdf;base64,AQID',
                },
                { type: 'input_text', text: 'Read this' },
            ],
        });
    });

    it('inlines a text attachment as text', async () => {
        const segments = [
            { role: PromptRole.user, content: 'Read this', files: [source('notes.txt', 'text/plain', [104, 105])] },
        ] as unknown as PromptSegment[];

        const prompt = await formatOpenAILikeMultimodalPrompt(segments, OPTIONS);

        expect(prompt[0]).toMatchObject({
            content: [
                { type: 'input_text', text: 'hi' },
                { type: 'input_text', text: 'Read this' },
            ],
        });
    });

    it('assumes an image when the attachment has no mime type', async () => {
        const segments = [
            { role: PromptRole.user, content: 'Look', files: [source('unknown', undefined)] },
        ] as unknown as PromptSegment[];

        const prompt = await formatOpenAILikeMultimodalPrompt(segments, OPTIONS);

        expect(prompt[0]).toMatchObject({
            content: [
                { type: 'input_image', image_url: 'data:image/jpeg;base64,AQID', detail: 'auto' },
                { type: 'input_text', text: 'Look' },
            ],
        });
    });
});
