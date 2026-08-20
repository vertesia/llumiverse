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
import type { VertexAIDriver } from '../index.js';
import { GeminiModelDefinition } from './gemini.js';

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
    it('inlines an image attached to a tool result as a functionResponse part', async () => {
        const contents = await createPrompt([toolSegment([source('https://signed.example/plot.png')])]);

        expect(contents[0].parts?.[0].functionResponse).toEqual({
            name: 'view_image',
            response: { artifact_path: 'out/plot.png' },
            parts: [{ inlineData: { data: 'AQID', mimeType: 'image/jpeg' } }],
        });
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
