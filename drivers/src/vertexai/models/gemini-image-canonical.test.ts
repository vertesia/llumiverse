import {
    type GenerateContentParameters,
    type GenerateContentResponse,
    type GoogleGenAI,
    Modality,
} from '@google/genai';
import type { ConversationDocument, ConversationStreamEvent } from '@llumiverse/conversation';
import type { CanonicalExecutionEventStream, ExecutionOptions } from '@llumiverse/core';
import { PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';

const MODEL = 'publishers/google/models/gemini-2.5-flash-image';
const IMAGE_BASE64 = 'iVBORw0KGgo=';
const IMAGE_HASH = 'sha256:4c4b6a3be1314ab86138bef4314dde022e600960d8689a2c8f8631802d20dab6';
const IMAGE_OUTPUT_MODALITY = 'image' as NonNullable<ExecutionOptions['output_modality']>;

class GeminiImageTestDriver extends VertexAIDriver {
    constructor(private readonly response: GenerateContentResponse) {
        super({ project: 'test-project', region: 'us-central1', geminiContextCache: false });
    }

    readonly generateContent = vi.fn(async (_request: GenerateContentParameters) => this.response);

    override getGoogleGenAIClient(): GoogleGenAI {
        return {
            models: {
                generateContent: this.generateContent,
            },
        } as unknown as GoogleGenAI;
    }
}

function options(attempt: string, conversation?: ConversationDocument): ExecutionOptions {
    return {
        model: MODEL,
        ...(conversation === undefined ? {} : { conversation }),
        output_modality: IMAGE_OUTPUT_MODALITY,
        model_options: {
            _option_id: 'vertexai-gemini',
            image_size: '1K',
            image_aspect_ratio: '1:1',
            output_mime_type: 'image/jpeg',
            output_compression_quality: 60,
        },
        conversation_runtime: {
            conversation_id: 'conversation:gemini-image',
            request_id: 'request:gemini-image',
            attempt_id: `attempt:gemini-image:${attempt}`,
            input_operation_id: 'input:gemini-image',
            response_operation_id: 'response:gemini-image',
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
            completed_at: '2026-09-30T00:00:00.000Z',
        },
    };
}

async function collect(stream: CanonicalExecutionEventStream): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('Gemini image canonical execution', () => {
    it('forwards image options and accepts verified native image bytes as a canonical asset', async () => {
        const driver = new GeminiImageTestDriver({
            responseId: 'gemini-image-response',
            modelVersion: 'gemini-2.5-flash-image',
            candidates: [
                {
                    finishReason: 'STOP',
                    content: {
                        role: 'model',
                        parts: [{ inlineData: { data: IMAGE_BASE64, mimeType: 'image/png' } }],
                    },
                },
            ],
            usageMetadata: {
                promptTokenCount: 4,
                candidatesTokenCount: 1290,
                totalTokenCount: 1294,
            },
        } as GenerateContentResponse);

        const result = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'Draw a small blue circle.' }],
            options('first'),
        );

        expect(driver.generateContent).toHaveBeenCalledOnce();
        expect(driver.generateContent).toHaveBeenCalledWith(
            expect.objectContaining({
                model: 'gemini-2.5-flash-image',
                config: expect.objectContaining({
                    responseModalities: [Modality.TEXT, Modality.IMAGE],
                    imageConfig: expect.objectContaining({
                        imageSize: '1K',
                        aspectRatio: '1:1',
                        outputMimeType: 'image/jpeg',
                        outputCompressionQuality: 60,
                    }),
                }),
            }),
        );
        expect(result.accepted_output.generation).toMatchObject({
            protocol: 'google.generate_content',
            requested_model: MODEL,
            resolved_model: 'gemini-2.5-flash-image',
        });
        expect(result.accepted_output.generation).not.toHaveProperty('request_receipt');
        expect(result.conversation.generations[result.accepted_output.generation.id]).toMatchObject({
            request_receipt: {
                target: {
                    provider: 'vertexai',
                    protocol: 'google.generate_content',
                    model: MODEL,
                },
            },
        });
        expect(result.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'image', asset_id: expect.any(String) }),
        );
        expect(Object.values(result.accepted_output.assets)).toEqual([
            expect.objectContaining({
                kind: 'image',
                mime_type: 'image/png',
                byte_length: 8,
                content_hash: IMAGE_HASH,
                storage: { type: 'inline_base64', data: IMAGE_BASE64 },
            }),
        ]);

        const persisted = JSON.parse(JSON.stringify(result.conversation)) as ConversationDocument;
        let retryPublishCount = 0;
        const retry = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Draw a small blue circle.' }],
            {
                ...options('retry', persisted),
                on_canonical_request_prepared: async () => {
                    retryPublishCount += 1;
                },
            },
            undefined,
            { stream_id: 'stream:gemini-image:retry' },
        );
        const events = await collect(retry);

        expect(events).toEqual([
            expect.objectContaining({ type: 'response_accepted', sequence: 0, origin: 'accepted_recovery' }),
        ]);
        expect(retry.completion?.accepted_output).toEqual(result.accepted_output);
        expect(retryPublishCount).toBe(0);
        expect(driver.generateContent).toHaveBeenCalledOnce();
        await retry.closed;
    });
});
