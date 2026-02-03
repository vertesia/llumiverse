/**
 * Unit tests for the fixOrphanedToolUse functions in Claude and Bedrock drivers.
 *
 * These functions handle the case where an agent is stopped mid-tool-execution,
 * leaving tool_use blocks without corresponding tool_result blocks.
 * The Anthropic/AWS APIs require every tool_use to have a matching tool_result.
 *
 * Bug context: When user stops an agent while tools are being executed,
 * the conversation has orphaned tool_use blocks. Sending a new message
 * without fixing this causes API errors like:
 * "tool_use ids were found without tool_result blocks immediately after"
 */

import { describe, expect, test } from 'vitest';
import { fixOrphanedToolUse as fixOrphanedToolUseClaude } from '../src/vertexai/models/claude.js';
import { fixOrphanedToolUse as fixOrphanedToolUseBedrock } from '../src/bedrock/index.js';
import { MessageParam } from '@anthropic-ai/sdk/resources/index.js';
import { Message } from '@aws-sdk/client-bedrock-runtime';

describe('fixOrphanedToolUse - Claude', () => {

    test('returns empty array for empty input', () => {
        const result = fixOrphanedToolUseClaude([]);
        expect(result).toEqual([]);
    });

    test('returns unchanged array when no orphaned tool_use', () => {
        const messages: MessageParam[] = [
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'Using a tool...' },
                    { type: 'tool_use', id: 'tool_1', name: 'search', input: { query: 'test' } },
                ]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'tool_1', content: 'Search result' }]
            },
        ];

        const result = fixOrphanedToolUseClaude(messages);
        expect(result).toEqual(messages);
    });

    test('injects synthetic tool_result for orphaned tool_use followed by text message', () => {
        const messages: MessageParam[] = [
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'Using a tool...' },
                    { type: 'tool_use', id: 'tool_1', name: 'search', input: { query: 'test' } },
                ]
            },
            {
                role: 'user',
                content: [{ type: 'text', text: 'Actually, stop that and do something else' }]
            },
        ];

        const result = fixOrphanedToolUseClaude(messages);

        expect(result).toHaveLength(2);
        expect(result[0].role).toBe('assistant');
        expect(result[1].role).toBe('user');

        // The user message should have synthetic tool_result prepended
        const userContent = result[1].content as Array<{ type: string; tool_use_id?: string; content?: string; text?: string }>;
        expect(userContent).toHaveLength(2);
        expect(userContent[0].type).toBe('tool_result');
        expect(userContent[0].tool_use_id).toBe('tool_1');
        expect(userContent[0].content).toContain('Tool interrupted');
        expect(userContent[0].content).toContain('search');
        expect(userContent[1].type).toBe('text');
        expect(userContent[1].text).toBe('Actually, stop that and do something else');
    });

    test('handles multiple orphaned tool_use blocks', () => {
        const messages: MessageParam[] = [
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'Using multiple tools...' },
                    { type: 'tool_use', id: 'tool_1', name: 'search', input: {} },
                    { type: 'tool_use', id: 'tool_2', name: 'fetch', input: {} },
                    { type: 'tool_use', id: 'tool_3', name: 'process', input: {} },
                ]
            },
            {
                role: 'user',
                content: [{ type: 'text', text: 'Stop!' }]
            },
        ];

        const result = fixOrphanedToolUseClaude(messages);

        const userContent = result[1].content as Array<{ type: string; tool_use_id?: string }>;
        expect(userContent).toHaveLength(4); // 3 synthetic tool_results + 1 text

        // Verify synthetic tool_results
        expect(userContent[0].type).toBe('tool_result');
        expect(userContent[0].tool_use_id).toBe('tool_1');
        expect(userContent[1].type).toBe('tool_result');
        expect(userContent[1].tool_use_id).toBe('tool_2');
        expect(userContent[2].type).toBe('tool_result');
        expect(userContent[2].tool_use_id).toBe('tool_3');
        expect(userContent[3].type).toBe('text');
    });

    test('handles partial tool_results - only injects for missing ones', () => {
        const messages: MessageParam[] = [
            {
                role: 'assistant',
                content: [
                    { type: 'tool_use', id: 'tool_1', name: 'search', input: {} },
                    { type: 'tool_use', id: 'tool_2', name: 'fetch', input: {} },
                ]
            },
            {
                role: 'user',
                content: [
                    { type: 'tool_result', tool_use_id: 'tool_1', content: 'Search completed' },
                    { type: 'text', text: 'Continue with this result' },
                ]
            },
        ];

        const result = fixOrphanedToolUseClaude(messages);

        const userContent = result[1].content as Array<{ type: string; tool_use_id?: string }>;
        expect(userContent).toHaveLength(3); // 1 synthetic + 1 real tool_result + 1 text

        // Synthetic tool_result for tool_2 should be first
        expect(userContent[0].type).toBe('tool_result');
        expect(userContent[0].tool_use_id).toBe('tool_2');
        // Original tool_result for tool_1
        expect(userContent[1].type).toBe('tool_result');
        expect(userContent[1].tool_use_id).toBe('tool_1');
    });

    test('handles string content in user message', () => {
        const messages: MessageParam[] = [
            {
                role: 'assistant',
                content: [
                    { type: 'tool_use', id: 'tool_1', name: 'search', input: {} },
                ]
            },
            {
                role: 'user',
                content: 'Stop and do something else'
            },
        ];

        const result = fixOrphanedToolUseClaude(messages);

        const userContent = result[1].content as Array<{ type: string; tool_use_id?: string; text?: string }>;
        expect(userContent).toHaveLength(2);
        expect(userContent[0].type).toBe('tool_result');
        expect(userContent[0].tool_use_id).toBe('tool_1');
        expect(userContent[1].type).toBe('text');
        expect(userContent[1].text).toBe('Stop and do something else');
    });

    test('real-world scenario: user stops agent mid-execution', () => {
        const messages: MessageParam[] = [
            // Initial conversation
            { role: 'user', content: [{ type: 'text', text: 'Search for documents' }] },
            {
                role: 'assistant',
                content: [{ type: 'text', text: 'I will search for documents.' }]
            },
            { role: 'user', content: [{ type: 'text', text: 'Yes, proceed' }] },
            // Assistant starts tool execution but user stops it
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'Searching...' },
                    { type: 'tool_use', id: 'toolu_search', name: 'search_documents', input: { query: 'important docs' } },
                    { type: 'tool_use', id: 'toolu_analyze', name: 'analyze_results', input: {} },
                ]
            },
            // User sends new message (agent was stopped, no tool_results)
            { role: 'user', content: [{ type: 'text', text: 'Never mind, do something else instead' }] },
        ];

        const result = fixOrphanedToolUseClaude(messages);

        expect(result).toHaveLength(5);

        // Check that the last user message has synthetic tool_results
        const lastUserContent = result[4].content as Array<{ type: string; tool_use_id?: string; content?: string }>;
        expect(lastUserContent).toHaveLength(3); // 2 synthetic + 1 text

        expect(lastUserContent[0].type).toBe('tool_result');
        expect(lastUserContent[0].tool_use_id).toBe('toolu_search');
        expect(lastUserContent[0].content).toContain('user stopped');

        expect(lastUserContent[1].type).toBe('tool_result');
        expect(lastUserContent[1].tool_use_id).toBe('toolu_analyze');
        expect(lastUserContent[1].content).toContain('user stopped');

        expect(lastUserContent[2].type).toBe('text');
    });
});

describe('fixOrphanedToolUse - Bedrock', () => {

    test('returns empty array for empty input', () => {
        const result = fixOrphanedToolUseBedrock([]);
        expect(result).toEqual([]);
    });

    test('returns unchanged array when no orphaned toolUse', () => {
        const messages: Message[] = [
            {
                role: 'assistant',
                content: [
                    { text: 'Using a tool...' },
                    { toolUse: { toolUseId: 'tool_1', name: 'search', input: { query: 'test' } } },
                ]
            },
            {
                role: 'user',
                content: [{ toolResult: { toolUseId: 'tool_1', content: [{ text: 'Search result' }] } }]
            },
        ];

        const result = fixOrphanedToolUseBedrock(messages);
        expect(result).toEqual(messages);
    });

    test('injects synthetic toolResult for orphaned toolUse followed by text message', () => {
        const messages: Message[] = [
            {
                role: 'assistant',
                content: [
                    { text: 'Using a tool...' },
                    { toolUse: { toolUseId: 'tool_1', name: 'search', input: { query: 'test' } } },
                ]
            },
            {
                role: 'user',
                content: [{ text: 'Actually, stop that and do something else' }]
            },
        ];

        const result = fixOrphanedToolUseBedrock(messages);

        expect(result).toHaveLength(2);
        expect(result[0].role).toBe('assistant');
        expect(result[1].role).toBe('user');

        // The user message should have synthetic toolResult prepended
        const userContent = result[1].content!;
        expect(userContent).toHaveLength(2);
        expect(userContent[0].toolResult).toBeDefined();
        expect(userContent[0].toolResult!.toolUseId).toBe('tool_1');
        expect(userContent[0].toolResult!.content![0].text).toContain('Tool interrupted');
        expect(userContent[0].toolResult!.content![0].text).toContain('search');
        expect(userContent[1].text).toBe('Actually, stop that and do something else');
    });

    test('handles multiple orphaned toolUse blocks', () => {
        const messages: Message[] = [
            {
                role: 'assistant',
                content: [
                    { text: 'Using multiple tools...' },
                    { toolUse: { toolUseId: 'tool_1', name: 'search', input: {} } },
                    { toolUse: { toolUseId: 'tool_2', name: 'fetch', input: {} } },
                    { toolUse: { toolUseId: 'tool_3', name: 'process', input: {} } },
                ]
            },
            {
                role: 'user',
                content: [{ text: 'Stop!' }]
            },
        ];

        const result = fixOrphanedToolUseBedrock(messages);

        const userContent = result[1].content!;
        expect(userContent).toHaveLength(4); // 3 synthetic toolResults + 1 text

        // Verify synthetic toolResults
        expect(userContent[0].toolResult?.toolUseId).toBe('tool_1');
        expect(userContent[1].toolResult?.toolUseId).toBe('tool_2');
        expect(userContent[2].toolResult?.toolUseId).toBe('tool_3');
        expect(userContent[3].text).toBe('Stop!');
    });

    test('handles partial toolResults - only injects for missing ones', () => {
        const messages: Message[] = [
            {
                role: 'assistant',
                content: [
                    { toolUse: { toolUseId: 'tool_1', name: 'search', input: {} } },
                    { toolUse: { toolUseId: 'tool_2', name: 'fetch', input: {} } },
                ]
            },
            {
                role: 'user',
                content: [
                    { toolResult: { toolUseId: 'tool_1', content: [{ text: 'Search completed' }] } },
                    { text: 'Continue with this result' },
                ]
            },
        ];

        const result = fixOrphanedToolUseBedrock(messages);

        const userContent = result[1].content!;
        expect(userContent).toHaveLength(3); // 1 synthetic + 1 real toolResult + 1 text

        // Synthetic toolResult for tool_2 should be first
        expect(userContent[0].toolResult?.toolUseId).toBe('tool_2');
        // Original toolResult for tool_1
        expect(userContent[1].toolResult?.toolUseId).toBe('tool_1');
    });

    test('real-world scenario: user stops agent mid-execution', () => {
        const messages: Message[] = [
            // Initial conversation
            { role: 'user', content: [{ text: 'Search for documents' }] },
            { role: 'assistant', content: [{ text: 'I will search for documents.' }] },
            { role: 'user', content: [{ text: 'Yes, proceed' }] },
            // Assistant starts tool execution but user stops it
            {
                role: 'assistant',
                content: [
                    { text: 'Searching...' },
                    { toolUse: { toolUseId: 'toolu_search', name: 'search_documents', input: { query: 'important docs' } } },
                    { toolUse: { toolUseId: 'toolu_analyze', name: 'analyze_results', input: {} } },
                ]
            },
            // User sends new message (agent was stopped, no toolResults)
            { role: 'user', content: [{ text: 'Never mind, do something else instead' }] },
        ];

        const result = fixOrphanedToolUseBedrock(messages);

        expect(result).toHaveLength(5);

        // Check that the last user message has synthetic toolResults
        const lastUserContent = result[4].content!;
        expect(lastUserContent).toHaveLength(3); // 2 synthetic + 1 text

        expect(lastUserContent[0].toolResult?.toolUseId).toBe('toolu_search');
        expect(lastUserContent[0].toolResult?.content![0].text).toContain('user stopped');

        expect(lastUserContent[1].toolResult?.toolUseId).toBe('toolu_analyze');
        expect(lastUserContent[1].toolResult?.content![0].text).toContain('user stopped');

        expect(lastUserContent[2].text).toBe('Never mind, do something else instead');
    });
});
