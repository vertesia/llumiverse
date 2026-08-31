import { type DataSource, type ExecutionOptions, PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import {
    anthropicUsageToTokenUsage,
    buildClaudeStreamingConversation,
    formatClaudePrompt,
    getClaudePayload,
    updateClaudeConversation,
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

    it('keeps one stable result-schema system block across agent tool continuations', async () => {
        const options: ExecutionOptions = {
            model: 'claude-sonnet-4-6',
            prompt_cache_key: 'agent-stable-prefix',
            result_schema: { type: 'object', properties: { value: { type: 'string' } } },
        };
        const initial = await formatClaudePrompt(
            [
                { role: PromptRole.system, content: 'agent instructions' },
                { role: PromptRole.user, content: 'start work' },
            ],
            options,
        );
        const continuation = await formatClaudePrompt(
            [{ role: PromptRole.tool, tool_use_id: 'tool-1', content: 'first result' }],
            options,
        );
        const nextContinuation = await formatClaudePrompt(
            [{ role: PromptRole.tool, tool_use_id: 'tool-2', content: 'second result' }],
            options,
        );

        const firstConversation = updateClaudeConversation(initial, continuation);
        const secondConversation = updateClaudeConversation(firstConversation, nextContinuation);
        const schemaBlocks = secondConversation.system?.filter((block) => block.text.includes('JSON Schema'));

        expect(initial.messages[0].content[0]).toEqual({ type: 'text', text: 'start work' });
        expect(firstConversation.system).toEqual(secondConversation.system);
        expect(schemaBlocks).toHaveLength(1);
        expect(schemaBlocks?.[0].text).toContain('"value"');
    });

    it('collapses legacy duplicate agent result-schema system blocks on the next continuation', async () => {
        const options: ExecutionOptions = {
            model: 'claude-sonnet-4-6',
            prompt_cache_key: 'agent-stable-prefix',
            result_schema: { type: 'object', properties: { value: { type: 'string' } } },
        };
        const continuation = await formatClaudePrompt(
            [{ role: PromptRole.tool, tool_use_id: 'tool-1', content: 'result' }],
            options,
        );
        const schemaBlock = continuation.system?.find((block) => block.text.includes('JSON Schema'));
        expect(schemaBlock).toBeDefined();
        if (!schemaBlock) throw new Error('Expected generated result-schema system block');
        const legacyConversation = {
            messages: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'start work' }] }],
            system: [
                { type: 'text' as const, text: 'agent instructions' },
                schemaBlock,
                { ...schemaBlock },
                { ...schemaBlock },
            ],
        };

        const repaired = updateClaudeConversation(legacyConversation, continuation);

        expect(repaired.system?.filter((block) => block.text.includes('JSON Schema'))).toHaveLength(1);
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

    it('maps private required-tool hints to native Claude tool choice', () => {
        const { payload } = getClaudePayload(
            {
                model: 'claude-haiku-4-5',
                tools: [{ name: 'write_artifact', input_schema: { type: 'object', properties: {} } }],
                model_options: {
                    _option_id: 'anthropic-claude',
                    tool_choice: 'required',
                    required_tool_name: 'write_artifact',
                    parallel_tool_calls: false,
                } as never,
            },
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'Act now.' }] }] },
        );

        expect(payload.tool_choice).toEqual({
            type: 'tool',
            name: 'write_artifact',
            disable_parallel_tool_use: true,
        });
    });

    it('preserves adaptive thinking for a forced Claude tool turn', () => {
        const { payload } = getClaudePayload(
            {
                model: 'claude-sonnet-4-6',
                tools: [{ name: 'write_artifact', input_schema: { type: 'object', properties: {} } }],
                model_options: {
                    _option_id: 'anthropic-claude',
                    effort: 'medium',
                    tool_choice: 'required',
                    required_tool_name: 'write_artifact',
                } as never,
            },
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'Act now.' }] }] },
        );

        expect(payload.tool_choice).toEqual({
            type: 'tool',
            name: 'write_artifact',
            disable_parallel_tool_use: false,
        });
        expect(payload.thinking).toEqual({ type: 'adaptive', display: 'omitted' });
        expect(payload.output_config).toEqual({ effort: 'medium' });
    });

    it('disables only manual extended thinking for forced Anthropic-compatible tool turns', () => {
        const { payload } = getClaudePayload(
            {
                model: 'claude-3-7-sonnet',
                tools: [{ name: 'write_artifact', input_schema: { type: 'object', properties: {} } }],
                model_options: {
                    _option_id: 'anthropic-claude',
                    thinking_budget_tokens: 8_000,
                    tool_choice: 'required',
                    required_tool_name: 'write_artifact',
                } as never,
            },
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'Act now.' }] }] },
        );

        expect(payload.tool_choice).toMatchObject({ type: 'tool', name: 'write_artifact' });
        expect(payload.thinking).toEqual({ type: 'disabled' });
    });

    it('keeps sampling parameters suppressed for a forced future-Claude tool turn', () => {
        const { payload } = getClaudePayload(
            {
                model: 'claude-fable-5',
                tools: [{ name: 'write_artifact', input_schema: { type: 'object', properties: {} } }],
                model_options: {
                    _option_id: 'anthropic-claude',
                    effort: 'medium',
                    temperature: 0.4,
                    top_p: 0.8,
                    top_k: 20,
                    tool_choice: 'required',
                    required_tool_name: 'write_artifact',
                } as never,
            },
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'Act now.' }] }] },
        );

        expect(payload.tool_choice).toMatchObject({ type: 'tool', name: 'write_artifact' });
        expect(payload.thinking).toEqual({ type: 'adaptive', display: 'omitted' });
        expect(payload.output_config).toEqual({ effort: 'medium' });
        expect(payload.temperature).toBeUndefined();
        expect(payload.top_p).toBeUndefined();
        expect(payload.top_k).toBeUndefined();
    });

    it('rejects forced tool choice for Claude Mythos preview turns', () => {
        expect(() =>
            getClaudePayload(
                {
                    model: 'claude-mythos-preview',
                    tools: [{ name: 'write_artifact', input_schema: { type: 'object', properties: {} } }],
                    model_options: {
                        _option_id: 'anthropic-claude',
                        effort: 'medium',
                        tool_choice: 'required',
                        required_tool_name: 'write_artifact',
                    } as never,
                },
                { messages: [{ role: 'user', content: [{ type: 'text', text: 'Act now.' }] }] },
            ),
        ).toThrowError(
            expect.objectContaining({
                name: 'ToolChoiceConfigurationError',
                retryable: false,
                code: 400,
                message: expect.stringContaining('does not support forced tool choice'),
            }),
        );
    });

    it('preserves adaptive thinking and forced tool choice for released Claude Mythos turns', () => {
        const { payload } = getClaudePayload(
            {
                model: 'claude-mythos-5',
                tools: [{ name: 'write_artifact', input_schema: { type: 'object', properties: {} } }],
                model_options: {
                    _option_id: 'anthropic-claude',
                    effort: 'medium',
                    tool_choice: 'required',
                    required_tool_name: 'write_artifact',
                } as never,
            },
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'Act now.' }] }] },
        );

        expect(payload.tool_choice).toMatchObject({ type: 'tool', name: 'write_artifact' });
        expect(payload.thinking).toEqual({ type: 'adaptive', display: 'omitted' });
        expect(payload.output_config).toEqual({ effort: 'medium' });
    });

    it('rejects a forced Claude tool turn when no tools are available', () => {
        expect(() =>
            getClaudePayload(
                {
                    model: 'claude-sonnet-4-6',
                    model_options: {
                        _option_id: 'anthropic-claude',
                        tool_choice: 'required',
                        required_tool_name: 'write_artifact',
                    } as never,
                },
                { messages: [{ role: 'user', content: [{ type: 'text', text: 'Act now.' }] }] },
            ),
        ).toThrowError(
            expect.objectContaining({
                name: 'ToolChoiceConfigurationError',
                retryable: false,
                code: 400,
            }),
        );
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
