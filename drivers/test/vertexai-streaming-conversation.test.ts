import type { Content } from '@google/genai';
import { type ExecutionOptions, unwrapConversationArray } from '@llumiverse/core';
import { describe, expect, test } from 'vitest';
import { VertexAIDriver, type VertexAIPrompt } from '../src/index.js';
import type { Tree } from './__helpers__/test-utils.js';

function extractTextParts(message: { parts?: Array<{ text?: string }> }): string[] {
    if (!message.parts) return [];
    return message.parts.map((p) => p.text ?? '').filter(Boolean);
}

describe('VertexAI streaming conversation rebuild', () => {
    test('Gemini streaming path does not duplicate history when prompt already merged', () => {
        const driver = new VertexAIDriver({ project: 'test', region: 'us-central1' });

        const summary: Content = { role: 'model', parts: [{ text: 'summary' }] };
        const userMsg: Content = { role: 'user', parts: [{ text: 'continue' }] };

        const prompt = {
            contents: [summary, userMsg],
        };

        const options = {
            model: 'publishers/google/models/gemini-2.5-flash',
            conversation: [summary],
        } as ExecutionOptions;

        const result = [{ type: 'text', value: 'response' }];

        const conversation = driver.buildStreamingConversation(
            prompt as unknown as VertexAIPrompt,
            result,
            undefined,
            options,
        );

        const unwrapped = unwrapConversationArray<Content>(conversation) ?? (conversation as Content[]);

        const summaryCount = unwrapped.filter(
            (m) => m.role === 'model' && extractTextParts(m).includes('summary'),
        ).length;

        expect(summaryCount).toBe(1);
        expect(unwrapped.length).toBe(3);
        expect(extractTextParts(unwrapped[2])).toEqual(['response']);
    });

    test('Claude streaming path includes existing history once', () => {
        const driver = new VertexAIDriver({ project: 'test', region: 'us-central1' });

        const existing = {
            messages: [{ role: 'assistant', content: [{ type: 'text', text: 'summary' }] }],
            system: [{ type: 'text', text: 'system' }],
        };

        const prompt = {
            messages: [{ role: 'user', content: [{ type: 'text', text: 'continue' }] }],
        };

        const options = {
            model: 'publishers/anthropic/models/claude-sonnet-4-5',
            conversation: existing,
        } as ExecutionOptions;

        const result = [{ type: 'text', value: 'response' }];

        const conversation = driver.buildStreamingConversation(
            prompt as unknown as VertexAIPrompt,
            result,
            undefined,
            options,
        ) as unknown as Tree;

        expect(conversation.system).toEqual(existing.system);
        expect(conversation.messages.length).toBe(3);
        expect(conversation.messages[0].content[0].text).toBe('summary');
        expect(conversation.messages[1].content[0].text).toBe('continue');
        expect(conversation.messages[2].content[0].text).toBe('response');
    });

    test('Claude streaming path keeps a cached prefix immutable across appended tool turns', () => {
        const driver = new VertexAIDriver({ project: 'test', region: 'us-central1' });
        const largeToolResult = `SOURCE:${'x'.repeat(12_000)}`;
        const existing = {
            messages: [
                { role: 'assistant', content: [{ type: 'tool_use', id: 'read-1', name: 'read', input: {} }] },
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'read-1', content: largeToolResult }] },
            ],
            system: [{ type: 'text', text: 'system' }],
        };
        const options = {
            model: 'publishers/anthropic/models/claude-sonnet-4-6',
            model_options: { cache_enabled: true },
            stripTextMaxTokens: 2_000,
            conversation: existing,
        } as ExecutionOptions;

        const first = driver.buildStreamingConversation(
            {
                messages: [{ role: 'assistant', content: [{ type: 'text', text: 'continue' }] }],
            } as unknown as VertexAIPrompt,
            [],
            [{ id: 'tool-2', tool_name: 'app_workspace_edit', tool_input: { path: 'src/App.tsx' } }],
            options,
        ) as unknown as Tree;
        const second = driver.buildStreamingConversation(
            {
                messages: [
                    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'tool-2', content: 'edited' }] },
                ],
            } as unknown as VertexAIPrompt,
            [],
            [{ id: 'tool-3', tool_name: 'app_workspace_typecheck', tool_input: {} }],
            { ...options, conversation: first },
        ) as unknown as Tree;

        const firstMessages = first.messages as unknown as unknown[];
        const secondMessages = second.messages as unknown as unknown[];
        expect(JSON.stringify(firstMessages)).toContain(largeToolResult);
        expect(secondMessages.slice(0, firstMessages.length)).toEqual(firstMessages);
        expect(JSON.stringify(secondMessages.slice(0, firstMessages.length))).toBe(JSON.stringify(firstMessages));
    });
});
