/**
 * Unit tests for the mergeConsecutiveUserMessages function in the Claude driver.
 *
 * This function is critical for ensuring that tool_result blocks are properly
 * grouped in a single user message, as required by Anthropic's API.
 *
 * Bug context: When multiple tools are called in parallel, each tool result
 * was being added as a separate user message. This caused API errors like:
 * "unexpected `tool_use_id` found in `tool_result` blocks"
 * because Anthropic expects all tool_results for a single assistant turn
 * to be in one user message.
 */

import { describe, expect, test } from 'vitest';
import { mergeConsecutiveUserMessages } from '../src/vertexai/models/claude.js';
import { MessageParam } from '@anthropic-ai/sdk/resources/index.js';

describe('mergeConsecutiveUserMessages', () => {

    test('returns empty array for empty input', () => {
        const result = mergeConsecutiveUserMessages([]);
        expect(result).toEqual([]);
    });

    test('returns unchanged array when no consecutive user messages', () => {
        const messages: MessageParam[] = [
            { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
            { role: 'assistant', content: [{ type: 'text', text: 'Hi there!' }] },
            { role: 'user', content: [{ type: 'text', text: 'How are you?' }] },
        ];

        const result = mergeConsecutiveUserMessages(messages);
        expect(result).toEqual(messages);
    });

    test('merges consecutive user messages with tool_result blocks', () => {
        const messages: MessageParam[] = [
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'Using tools...' },
                    { type: 'tool_use', id: 'tool_1', name: 'search', input: { query: 'test' } },
                    { type: 'tool_use', id: 'tool_2', name: 'fetch', input: { url: 'http://example.com' } },
                ]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'tool_1', content: 'Search result' }]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'tool_2', content: 'Fetch result' }]
            },
        ];

        const result = mergeConsecutiveUserMessages(messages);

        expect(result).toHaveLength(2);
        expect(result[0].role).toBe('assistant');
        expect(result[1].role).toBe('user');

        // The user message should contain both tool_result blocks
        const userContent = result[1].content as Array<{ type: string; tool_use_id?: string }>;
        expect(userContent).toHaveLength(2);
        expect(userContent[0].type).toBe('tool_result');
        expect(userContent[0].tool_use_id).toBe('tool_1');
        expect(userContent[1].type).toBe('tool_result');
        expect(userContent[1].tool_use_id).toBe('tool_2');
    });

    test('merges three consecutive tool_result messages', () => {
        const messages: MessageParam[] = [
            {
                role: 'assistant',
                content: [
                    { type: 'tool_use', id: 'tool_a', name: 'tool_a', input: {} },
                    { type: 'tool_use', id: 'tool_b', name: 'tool_b', input: {} },
                    { type: 'tool_use', id: 'tool_c', name: 'tool_c', input: {} },
                ]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'tool_a', content: 'Result A' }]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'tool_b', content: 'Result B' }]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'tool_c', content: 'Result C' }]
            },
        ];

        const result = mergeConsecutiveUserMessages(messages);

        expect(result).toHaveLength(2);

        const userContent = result[1].content as Array<{ type: string; tool_use_id?: string }>;
        expect(userContent).toHaveLength(3);
        expect(userContent.map(c => c.tool_use_id)).toEqual(['tool_a', 'tool_b', 'tool_c']);
    });

    test('handles multiple separate groups of consecutive user messages', () => {
        const messages: MessageParam[] = [
            // First tool call and results
            {
                role: 'assistant',
                content: [
                    { type: 'tool_use', id: 'tool_1', name: 'search', input: {} },
                    { type: 'tool_use', id: 'tool_2', name: 'fetch', input: {} },
                ]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'tool_1', content: 'Result 1' }]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'tool_2', content: 'Result 2' }]
            },
            // Second tool call and results
            {
                role: 'assistant',
                content: [
                    { type: 'tool_use', id: 'tool_3', name: 'process', input: {} },
                    { type: 'tool_use', id: 'tool_4', name: 'save', input: {} },
                ]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'tool_3', content: 'Result 3' }]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'tool_4', content: 'Result 4' }]
            },
        ];

        const result = mergeConsecutiveUserMessages(messages);

        expect(result).toHaveLength(4);
        expect(result[0].role).toBe('assistant');
        expect(result[1].role).toBe('user');
        expect(result[2].role).toBe('assistant');
        expect(result[3].role).toBe('user');

        // First group of tool results
        const userContent1 = result[1].content as Array<{ type: string; tool_use_id?: string }>;
        expect(userContent1).toHaveLength(2);
        expect(userContent1.map(c => c.tool_use_id)).toEqual(['tool_1', 'tool_2']);

        // Second group of tool results
        const userContent2 = result[3].content as Array<{ type: string; tool_use_id?: string }>;
        expect(userContent2).toHaveLength(2);
        expect(userContent2.map(c => c.tool_use_id)).toEqual(['tool_3', 'tool_4']);
    });

    test('handles string content in user messages', () => {
        const messages: MessageParam[] = [
            { role: 'user', content: 'Hello' },
            { role: 'user', content: 'World' },
        ];

        const result = mergeConsecutiveUserMessages(messages);

        expect(result).toHaveLength(1);
        expect(result[0].role).toBe('user');

        const content = result[0].content as Array<{ type: string; text?: string }>;
        expect(content).toHaveLength(2);
        expect(content[0]).toEqual({ type: 'text', text: 'Hello' });
        expect(content[1]).toEqual({ type: 'text', text: 'World' });
    });

    test('handles mixed array and string content', () => {
        const messages: MessageParam[] = [
            { role: 'user', content: 'First message' },
            { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'tool_1', content: 'Result' }] },
        ];

        const result = mergeConsecutiveUserMessages(messages);

        expect(result).toHaveLength(1);

        const content = result[0].content as Array<{ type: string }>;
        expect(content).toHaveLength(2);
        expect(content[0].type).toBe('text');
        expect(content[1].type).toBe('tool_result');
    });

    test('preserves single user message unchanged', () => {
        const messages: MessageParam[] = [
            { role: 'assistant', content: [{ type: 'text', text: 'Response' }] },
            {
                role: 'user',
                content: [
                    { type: 'tool_result', tool_use_id: 'tool_1', content: 'Result' },
                    { type: 'tool_result', tool_use_id: 'tool_2', content: 'Result 2' },
                ]
            },
        ];

        const result = mergeConsecutiveUserMessages(messages);

        // No merging needed, should return same structure
        expect(result).toHaveLength(2);
        expect(result).toEqual(messages);
    });

    test('real-world scenario: parallel tool execution with checkpointing', () => {
        // This simulates the exact bug scenario from the error report
        const messages: MessageParam[] = [
            // Initial user request
            { role: 'user', content: [{ type: 'text', text: 'Test the tools' }] },
            // Assistant uses think tool
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'Planning...' },
                    { type: 'tool_use', id: 'toolu_think', name: 'think', input: { thought: 'Planning' } },
                ]
            },
            // Think result
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'toolu_think', content: 'Thought recorded.' }]
            },
            // Assistant uses 3 tools in parallel
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'Running tests...' },
                    { type: 'tool_use', id: 'toolu_search', name: 'search_documents', input: { query: 'test' } },
                    { type: 'tool_use', id: 'toolu_shell', name: 'execute_shell', input: { command: 'echo test' } },
                    { type: 'tool_use', id: 'toolu_time', name: 'learn_current_datetime', input: {} },
                ]
            },
            // 3 separate tool result messages (the bug!)
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'toolu_search', content: 'Search results...' }]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'toolu_shell', content: 'Shell output...' }]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'toolu_time', content: 'Time info...' }]
            },
            // Assistant uses 2 more tools
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'Creating document...' },
                    { type: 'tool_use', id: 'toolu_doc', name: 'create_document', input: { name: 'test' } },
                    { type: 'tool_use', id: 'toolu_artifact', name: 'write_artifact', input: { name: 'test.py' } },
                ]
            },
            // 2 more separate tool result messages
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'toolu_doc', content: 'Document created.' }]
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'toolu_artifact', content: 'Artifact written.' }]
            },
        ];

        const result = mergeConsecutiveUserMessages(messages);

        // Should have: user, assistant, user, assistant, user, assistant, user
        expect(result).toHaveLength(7);

        // Verify the structure is correct
        expect(result[0].role).toBe('user');
        expect(result[1].role).toBe('assistant');
        expect(result[2].role).toBe('user');  // think result
        expect(result[3].role).toBe('assistant');
        expect(result[4].role).toBe('user');  // merged: search + shell + time results
        expect(result[5].role).toBe('assistant');
        expect(result[6].role).toBe('user');  // merged: doc + artifact results

        // Verify the merged tool results
        const toolResults1 = result[4].content as Array<{ type: string; tool_use_id?: string }>;
        expect(toolResults1).toHaveLength(3);
        expect(toolResults1.map(c => c.tool_use_id)).toEqual(['toolu_search', 'toolu_shell', 'toolu_time']);

        const toolResults2 = result[6].content as Array<{ type: string; tool_use_id?: string }>;
        expect(toolResults2).toHaveLength(2);
        expect(toolResults2.map(c => c.tool_use_id)).toEqual(['toolu_doc', 'toolu_artifact']);
    });
});
