import type { MessageParam } from '@anthropic-ai/sdk/resources/messages.js';
import type { ExecutionOptions } from '@llumiverse/common';
import { parseConversationDocument } from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import type { OpenAIChatCompletionsPrompt } from '../openai/openai_chat_completions.js';
import {
    compileOpenAIChatCompletionsConversation,
    exportLegacyOpenAIChatCompletionsConversation,
    prepareOpenAIChatCanonicalState,
} from '../openai/openai-chat-conversation-adapter.js';
import type { ClaudePrompt } from '../shared/claude-messages.js';
import {
    compileClaudeMessagesConversation,
    exportLegacyClaudeMessagesConversation,
    prepareClaudeCanonicalState,
} from '../shared/claude-messages-conversation-adapter.js';
import { exportLegacyConversation } from './index.js';

const recordedAt = '2026-09-11T00:00:00.000Z';

function executionOptions(model: string, conversationId: string): ExecutionOptions {
    return {
        model,
        conversation_runtime: {
            conversation_id: conversationId,
            request_id: `request:${conversationId}`,
            attempt_id: `attempt:${conversationId}`,
            input_operation_id: `input:${conversationId}`,
            response_operation_id: `response:${conversationId}`,
            recorded_at: recordedAt,
        },
    };
}

async function importOpenAI(history: OpenAIChatCompletionsPrompt, conversationId: string) {
    return prepareOpenAIChatCanonicalState({
        conversation: history,
        prompt: { _is_openai_chat_completions: true, messages: [] },
        options: executionOptions('gpt-test', conversationId),
        provider: 'openai',
    });
}

async function importClaude(history: ClaudePrompt, conversationId: string) {
    return prepareClaudeCanonicalState({
        conversation: history,
        prompt: { messages: [] },
        options: executionOptions('claude-test', conversationId),
        provider: 'anthropic',
    });
}

describe('canonical native adapter conformance', () => {
    it('preserves OpenAI reasoning, raw tool arguments, image detail, and native call identities', async () => {
        const history: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'look exactly' },
                        { type: 'image_url', image_url: { url: 'data:image/png;base64,YWJj', detail: 'high' } },
                    ],
                },
                {
                    role: 'assistant',
                    content: 'checking',
                    reasoning_content: 'private reasoning',
                    tool_calls: [
                        {
                            id: 'call-exact',
                            type: 'function',
                            function: { name: 'lookup', arguments: '{ "city" : "Tokyo" }' },
                        },
                    ],
                },
                { role: 'tool', tool_call_id: 'call-exact', content: 'sunny' },
            ],
        };
        const state = await prepareOpenAIChatCanonicalState({
            conversation: history,
            prompt: { _is_openai_chat_completions: true, messages: [] },
            options: {
                ...executionOptions('gpt-test', 'openai-fidelity'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
            provider: 'openai',
        });

        expect(parseConversationDocument(state.document)).toEqual(state.document);
        expect(exportLegacyOpenAIChatCompletionsConversation(state.document).messages).toEqual(history.messages);
        expect(exportLegacyConversation(state.document)).toEqual(
            exportLegacyOpenAIChatCompletionsConversation(state.document),
        );
        const toolResult = state.document.turns.find((turn) => turn.kind === 'tool');
        expect(toolResult?.blocks[0]).toMatchObject({ call_id: 'call-exact', status: 'unknown' });
    });

    it('preserves Claude signed and redacted blocks and marks imported status without evidence unknown', async () => {
        const messages: MessageParam[] = [
            { role: 'user', content: [{ type: 'text', text: 'question' }] },
            {
                role: 'assistant',
                content: [
                    { type: 'thinking', thinking: 'plan', signature: 'signed-plan' },
                    { type: 'redacted_thinking', data: 'opaque-redaction' },
                    { type: 'text', text: 'answer' },
                    { type: 'tool_use', id: 'call-exact', name: 'lookup', input: { city: 'Tokyo' } },
                ],
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'call-exact', content: 'sunny' }],
            },
        ];
        const history = { messages } as ClaudePrompt;
        const state = await prepareClaudeCanonicalState({
            conversation: history,
            prompt: { messages: [] },
            options: {
                ...executionOptions('claude-test', 'claude-fidelity'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
            provider: 'anthropic',
        });

        const projection = exportLegacyClaudeMessagesConversation(state.document);
        expect(projection.messages[1]).toEqual(messages[1]);
        const toolResult = state.document.turns.find((turn) => turn.kind === 'tool');
        expect(toolResult?.blocks[0]).toMatchObject({ call_id: 'call-exact', status: 'unknown' });
    });

    it('transfers plain text and image media across protocols without losing payload bytes', async () => {
        const claude = await importClaude(
            {
                messages: [
                    {
                        role: 'user',
                        content: [
                            { type: 'text', text: 'caption' },
                            {
                                type: 'image',
                                source: { type: 'base64', media_type: 'image/png', data: 'YWJj' },
                            },
                        ],
                    },
                ],
            },
            'cross-media',
        );

        const openAIProjection = compileOpenAIChatCompletionsConversation(claude.document).conversation;
        expect(openAIProjection.messages).toEqual([
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'caption' },
                    { type: 'image_url', image_url: { url: 'data:image/png;base64,YWJj' } },
                ],
            },
        ]);
        expect(compileClaudeMessagesConversation(claude.document).conversation.messages[0]).toEqual({
            role: 'user',
            content: [
                { type: 'text', text: 'caption' },
                { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'YWJj' } },
            ],
        });
    });

    it('fails closed for protected foreign replay and visible unsupported media', async () => {
        const openAI = await importOpenAI(
            {
                _is_openai_chat_completions: true,
                messages: [{ role: 'assistant', content: 'answer', reasoning_content: 'provider reasoning' }],
            },
            'foreign-openai-replay',
        );
        expect(() => compileClaudeMessagesConversation(openAI.document)).toThrow(
            'cannot discard protected openai.chat.completions replay',
        );

        const claudeThinking = await importClaude(
            {
                messages: [
                    {
                        role: 'assistant',
                        content: [{ type: 'thinking', thinking: 'plan', signature: 'signature' }],
                    },
                ],
            },
            'foreign-claude-replay',
        );
        expect(() => compileOpenAIChatCompletionsConversation(claudeThinking.document)).toThrow(
            'cannot discard protected anthropic.messages replay',
        );

        const claudeDocument = await importClaude(
            {
                messages: [
                    {
                        role: 'user',
                        content: [
                            {
                                type: 'document',
                                source: { type: 'base64', media_type: 'application/pdf', data: 'YWJj' },
                            },
                        ],
                    },
                ],
            },
            'unsupported-document',
        );
        expect(() => compileOpenAIChatCompletionsConversation(claudeDocument.document)).toThrow(
            'cannot project canonical document block',
        );
    });
});
