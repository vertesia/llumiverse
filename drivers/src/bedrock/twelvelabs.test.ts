import { type DataSource, PromptRole } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { formatTwelvelabsPegasusPrompt, pegasusTokenUsage } from './twelvelabs.js';

function video(url: string): DataSource {
    return {
        name: 'clip.mp4',
        mime_type: 'video/mp4',
        getStream: async () =>
            new ReadableStream<Uint8Array>({
                start(controller) {
                    controller.enqueue(new Uint8Array([1, 2, 3]));
                    controller.close();
                },
            }),
        getURL: async () => url,
        getURI: async () => url,
    };
}

describe('TwelveLabs Pegasus request', () => {
    it('sends the result schema directly under responseFormat.jsonSchema', async () => {
        const schema = { type: 'object' as const, properties: { color: { type: 'string' as const } } };
        const request = await formatTwelvelabsPegasusPrompt(
            [{ role: PromptRole.user, content: 'Colour?', files: [video('https://example.com/clip.mp4')] }],
            {
                model: 'us.twelvelabs.pegasus-1-5-v1:0',
                result_schema: schema,
                model_options: { _option_id: 'bedrock-twelvelabs-pegasus', max_tokens: 600, temperature: 0 },
            },
        );

        expect(request).toEqual({
            inputPrompt: 'Colour?',
            mediaSource: { base64String: 'AQID' },
            temperature: 0,
            maxOutputTokens: 600,
            responseFormat: { type: 'json_schema', jsonSchema: schema },
        });
    });

    it('omits responseFormat.type for Pegasus 1.2, which rejects it', async () => {
        const schema = { type: 'object' as const };
        const request = await formatTwelvelabsPegasusPrompt(
            [{ role: PromptRole.user, content: 'Colour?', files: [video('https://example.com/clip.mp4')] }],
            { model: 'us.twelvelabs.pegasus-1-2-v1:0', result_schema: schema },
        );

        expect(request.responseFormat).toEqual({ jsonSchema: schema });
    });

    it('reports Pegasus token counts', () => {
        expect(pegasusTokenUsage({ message: 'Red', stopReason: 'stop', input_tokens: 212, output_tokens: 6 })).toEqual({
            prompt: 212,
            result: 6,
            total: 218,
        });
        expect(pegasusTokenUsage({ message: 'Red', finishReason: 'stop' })).toBeUndefined();
    });

    it('passes S3 video sources by URI', async () => {
        const request = await formatTwelvelabsPegasusPrompt(
            [{ role: PromptRole.user, content: 'Describe', files: [video('s3://bucket/clip.mp4')] }],
            { model: 'twelvelabs.pegasus-1-2-v1:0' },
        );

        expect(request.mediaSource).toEqual({ s3Location: { uri: 's3://bucket/clip.mp4' } });
        expect(request).not.toHaveProperty('responseFormat');
    });
});
