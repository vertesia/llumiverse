import { type DataSource, type ExecutionOptions, PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import {
    anthropicUsageToTokenUsage,
    buildClaudeStreamingConversation,
    formatClaudePrompt,
    getClaudePayload,
} from './claude-messages.js';

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

    it('keeps agent cache breakpoints on fixed blocks as the conversation grows', () => {
        const messages = Array.from({ length: 49 }, (_, index) => ({
            role: (index % 2 === 0 ? 'user' : 'assistant') as 'user' | 'assistant',
            content: [{ type: 'text' as const, text: `block-${index}` }],
        }));
        const options: ExecutionOptions = {
            model: 'claude-sonnet-4-6',
            prompt_cache_key: 'agent-stable-prefix',
            model_options: { cache_ttl: '1h' } as never,
        };

        const first = getClaudePayload(options, {
            system: [{ type: 'text', text: 'system' }],
            messages: messages.slice(0, 25),
        }).payload;
        const second = getClaudePayload(options, {
            system: [{ type: 'text', text: 'system' }],
            messages,
        }).payload;

        expect(second.messages.slice(0, first.messages.length)).toEqual(first.messages);
        expect(second.system).toEqual([{ type: 'text', text: 'system' }]);
        expect(
            second.messages.flatMap((message, messageIndex) =>
                Array.isArray(message.content)
                    ? message.content
                          .map((block) => ('cache_control' in block ? messageIndex : undefined))
                          .filter((index): index is number => index !== undefined)
                    : [],
            ),
        ).toEqual([0, 12, 24, 36]);
        expect(second.messages[0].content[0]).toMatchObject({
            cache_control: { type: 'ephemeral', ttl: '1h' },
        });
    });

    it('appends a user turn when the conversation ends with an assistant message on no-prefill models', () => {
        const options: ExecutionOptions = { model: 'claude-fable-5' };
        const prompt = {
            system: undefined,
            messages: [
                { role: 'user' as const, content: [{ type: 'text' as const, text: 'do the task' }] },
                { role: 'assistant' as const, content: [{ type: 'text' as const, text: 'partial answer' }] },
            ],
        };

        const { payload } = getClaudePayload(options, prompt);

        const last = payload.messages[payload.messages.length - 1];
        expect(last).toEqual({ role: 'user', content: [{ type: 'text', text: 'Continue.' }] });
    });

    it('preserves intentional assistant prefill on pre-4.6 models', () => {
        const options: ExecutionOptions = { model: 'claude-3-5-sonnet-20241022' };
        const prompt = {
            system: undefined,
            messages: [
                { role: 'user' as const, content: [{ type: 'text' as const, text: 'answer as JSON' }] },
                { role: 'assistant' as const, content: [{ type: 'text' as const, text: '{' }] },
            ],
        };

        const { payload } = getClaudePayload(options, prompt);

        const last = payload.messages[payload.messages.length - 1];
        expect(last?.role).toBe('assistant');
    });

    it('does not touch conversations already ending with a user turn', () => {
        const options: ExecutionOptions = { model: 'claude-fable-5' };
        const prompt = {
            system: undefined,
            messages: [
                { role: 'assistant' as const, content: [{ type: 'text' as const, text: 'previous reply' }] },
                { role: 'user' as const, content: [{ type: 'text' as const, text: 'next instruction' }] },
            ],
        };

        const { payload } = getClaudePayload(options, prompt);

        expect(payload.messages).toHaveLength(2);
        expect(payload.messages[payload.messages.length - 1]?.role).toBe('user');
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

    it('keeps the serialized cached conversation prefix immutable as tool turns are appended', () => {
        const largeHistoricalResult = `SOURCE:${'x'.repeat(12_000)}`;
        const historicalMessages = Array.from({ length: 14 }, (_, index) => ({
            role: (index % 2 === 0 ? 'assistant' : 'user') as 'assistant' | 'user',
            content: [
                index === 1
                    ? {
                          type: 'tool_result' as const,
                          tool_use_id: 'historical-read',
                          content: largeHistoricalResult,
                      }
                    : { type: 'text' as const, text: `historical-${index}` },
            ],
        }));
        const options: ExecutionOptions = {
            model: 'claude-sonnet-4-6',
            model_options: { cache_enabled: true } as never,
            stripTextMaxTokens: 8_000,
            conversation: { messages: historicalMessages },
        };

        const first = buildClaudeStreamingConversation(
            {
                messages: [
                    {
                        role: 'user',
                        content: [{ type: 'text', text: 'first new tool result' }],
                    },
                ],
            },
            [],
            [{ id: 'tool-1', tool_name: 'app_workspace_edit', tool_input: { path: 'src/App.tsx' } }],
            options,
        );
        const second = buildClaudeStreamingConversation(
            {
                messages: [
                    {
                        role: 'user',
                        content: [{ type: 'text', text: 'second new tool result' }],
                    },
                ],
            },
            [],
            [{ id: 'tool-2', tool_name: 'app_workspace_typecheck', tool_input: {} }],
            { ...options, conversation: first },
        );

        expect(JSON.stringify(first.messages)).toContain(largeHistoricalResult);
        expect(second.messages.slice(0, first.messages.length)).toEqual(first.messages);
        expect(JSON.stringify(second.messages.slice(0, first.messages.length))).toBe(JSON.stringify(first.messages));
    });

    it('retains sliding text truncation when prompt caching is disabled', () => {
        const largeHistoricalResult = `SOURCE:${'x'.repeat(12_000)}`;
        const conversation = buildClaudeStreamingConversation(
            {
                messages: Array.from({ length: 14 }, (_, index) => ({
                    role: (index % 2 === 0 ? 'assistant' : 'user') as 'assistant' | 'user',
                    content: [
                        index === 1
                            ? {
                                  type: 'tool_result' as const,
                                  tool_use_id: 'historical-read',
                                  content: largeHistoricalResult,
                              }
                            : { type: 'text' as const, text: `historical-${index}` },
                    ],
                })),
            },
            [],
            [{ id: 'tool-1', tool_name: 'app_workspace_edit', tool_input: { path: 'src/App.tsx' } }],
            {
                model: 'claude-sonnet-4-6',
                stripTextMaxTokens: 8_000,
            },
        );

        expect(JSON.stringify(conversation.messages)).not.toContain(largeHistoricalResult);
        expect(JSON.stringify(conversation.messages)).toContain('[Content truncated');
    });
});
