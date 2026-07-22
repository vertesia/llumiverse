import { type DataSource, type ExecutionOptions, PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { anthropicUsageToTokenUsage, formatClaudePrompt, getClaudePayload } from './claude-messages.js';

describe('formatClaudePrompt', () => {
    const imageSource = (): DataSource => ({
        name: 'page.jpg',
        mime_type: 'image/jpeg',
        getStream: vi.fn().mockResolvedValue(
            new ReadableStream({
                start(controller) {
                    controller.enqueue(new Uint8Array([1, 2, 3]));
                    controller.close();
                },
            }),
        ),
        getURL: vi.fn(),
        getURI: vi.fn(),
    });

    it('warns and skips video attachments', async () => {
        const getStream = vi.fn();
        const warn = vi.fn();
        const segments = [
            {
                role: PromptRole.user,
                content: 'Look at this',
                files: [
                    {
                        name: 'clip.mp4',
                        mime_type: 'video/mp4',
                        getStream,
                    },
                ],
            },
        ] as unknown as PromptSegment[];

        const prompt = await formatClaudePrompt(segments, { model: 'claude-haiku-4-5' } as never, { warn });

        expect(prompt.messages).toEqual([
            {
                role: 'user',
                content: [{ type: 'text', text: 'Look at this' }],
            },
        ]);
        expect(getStream).not.toHaveBeenCalled();
        expect(warn).toHaveBeenCalledWith(
            { file_name: 'clip.mp4', mime_type: 'video/mp4' },
            '[Claude] Skipping unsupported video attachment',
        );
    });

    it('places a routed cache breakpoint after stable source attachments and before the final task', async () => {
        const options: ExecutionOptions = {
            model: 'claude-sonnet-4-6',
            prompt_cache_key: 'document-prefix',
            result_schema: { type: 'object', properties: { value: { type: 'string' } } },
        };
        const prompt = await formatClaudePrompt(
            [
                { role: PromptRole.user, content: 'stable document source', files: [imageSource()] },
                { role: PromptRole.user, content: 'dynamic extraction task' },
            ],
            options,
        );

        const { payload } = getClaudePayload(options, prompt);

        expect(payload.messages).toEqual([
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'stable document source' },
                    {
                        type: 'image',
                        source: { type: 'base64', media_type: 'image/jpeg', data: 'AQID' },
                        cache_control: { type: 'ephemeral' },
                    },
                    { type: 'text', text: expect.stringContaining('dynamic extraction task') },
                ],
            },
        ]);
        expect(payload.messages[0].content[2]).toMatchObject({ text: expect.stringContaining('"value"') });
    });

    it('keeps the routed source prefix stable across different tasks and schemas', async () => {
        const createPayload = async (task: string, field: string) => {
            const options: ExecutionOptions = {
                model: 'claude-sonnet-4-6',
                prompt_cache_key: 'document-prefix',
                result_schema: { type: 'object', properties: { [field]: { type: 'string' } } },
            };
            const prompt = await formatClaudePrompt(
                [
                    { role: PromptRole.system, content: 'shared system' },
                    { role: PromptRole.user, content: 'stable document source' },
                    { role: PromptRole.user, content: task },
                ],
                options,
            );
            return getClaudePayload(options, prompt).payload;
        };

        const extraction = await createPayload('extract fields', 'invoice_number');
        const review = await createPayload('review fields', 'review_verdict');
        const extractionContent = extraction.messages[0].content;
        const reviewContent = review.messages[0].content;

        expect(extraction.system).toEqual(review.system);
        expect(Array.isArray(extractionContent) ? extractionContent[0] : undefined).toEqual(
            Array.isArray(reviewContent) ? reviewContent[0] : undefined,
        );
        expect(JSON.stringify(extraction.system)).not.toContain('invoice_number');
        expect(JSON.stringify(review.system)).not.toContain('review_verdict');
        expect(Array.isArray(extractionContent) ? extractionContent[1] : undefined).toMatchObject({
            type: 'text',
            text: expect.stringContaining('invoice_number'),
        });
        expect(Array.isArray(reviewContent) ? reviewContent[1] : undefined).toMatchObject({
            type: 'text',
            text: expect.stringContaining('review_verdict'),
        });
    });

    it('preserves model-option cache controls when no routing identity is supplied', () => {
        const options: ExecutionOptions = {
            model: 'claude-sonnet-4-6',
            model_options: { _option_id: 'anthropic-claude', cache_enabled: true } as never,
        };
        const prompt = {
            system: [{ type: 'text' as const, text: 'stable system prompt' }],
            messages: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'task' }] }],
        };

        const { payload } = getClaudePayload(options, prompt);

        expect(payload.system).toEqual([
            {
                type: 'text',
                text: 'stable system prompt',
                cache_control: { type: 'ephemeral' },
            },
        ]);
    });

    it('reports Claude cache reads and writes consistently for direct, Vertex, and Bedrock Mantle clients', () => {
        expect(
            anthropicUsageToTokenUsage({
                input_tokens: 25,
                output_tokens: 10,
                cache_read_input_tokens: 100,
                cache_creation_input_tokens: 50,
            }),
        ).toEqual({
            prompt: 175,
            prompt_new: 25,
            prompt_cached: 100,
            prompt_cache_write: 50,
            result: 10,
            total: 185,
        });
    });
});
