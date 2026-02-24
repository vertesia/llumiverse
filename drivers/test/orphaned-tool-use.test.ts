/**
 * Unit tests for the fixOrphanedToolUse functions in Claude, Bedrock, and OpenAI drivers.
 *
 * These functions handle the case where an agent is stopped mid-tool-execution,
 * leaving tool_use/function_call blocks without corresponding tool_result/function_call_output blocks.
 * The LLM APIs require every tool call to have a matching result.
 *
 * Bug context: When user stops an agent while tools are being executed,
 * the conversation has orphaned tool calls. Sending a new message
 * without fixing this causes API errors like:
 * - Anthropic: "tool_use ids were found without tool_result blocks immediately after"
 * - OpenAI: "No tool output found for function call <id>"
 */

import { describe, expect, test } from 'vitest';
import { fixOrphanedToolUse as fixOrphanedToolUseClaude } from '../src/vertexai/models/claude.js';
import { fixOrphanedToolUse as fixOrphanedToolUseBedrock } from '../src/bedrock/index.js';
import { fixOrphanedToolUse as fixOrphanedToolUseOpenAI } from '../src/openai/index.js';
import { MessageParam } from '@anthropic-ai/sdk/resources/index.js';
import { Message } from '@aws-sdk/client-bedrock-runtime';
import type OpenAI from 'openai';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;

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

describe('fixOrphanedToolUse - OpenAI', () => {

    test('returns empty array for empty input', () => {
        const result = fixOrphanedToolUseOpenAI([]);
        expect(result).toEqual([]);
    });

    test('returns single item unchanged', () => {
        const items: ResponseInputItem[] = [
            { role: 'user', content: 'Hello' },
        ];
        const result = fixOrphanedToolUseOpenAI(items);
        expect(result).toEqual(items);
    });

    test('returns unchanged array when no orphaned function_call', () => {
        const items: ResponseInputItem[] = [
            { role: 'user', content: 'Search for something' },
            { role: 'assistant', content: 'I will search.' },
            { type: 'function_call', call_id: 'fc_1', name: 'search', arguments: '{"query":"test"}' } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc_1', output: 'Search result' } as ResponseInputItem,
            { role: 'assistant', content: 'Here are the results.' },
        ];

        const result = fixOrphanedToolUseOpenAI(items);
        expect(result).toEqual(items);
    });

    test('injects synthetic function_call_output for orphaned function_call followed by user message', () => {
        const items: ResponseInputItem[] = [
            { role: 'assistant', content: 'Using a tool...' },
            { type: 'function_call', call_id: 'fc_1', name: 'ask_user', arguments: '{"question":"What?"}' } as ResponseInputItem,
            { role: 'user', content: 'Actually, stop that' },
        ];

        const result = fixOrphanedToolUseOpenAI(items);

        expect(result).toHaveLength(4);
        expect(result[0]).toEqual({ role: 'assistant', content: 'Using a tool...' });
        expect(result[1]).toEqual({ type: 'function_call', call_id: 'fc_1', name: 'ask_user', arguments: '{"question":"What?"}' });

        // Synthetic function_call_output should be injected before user message
        const synthetic = result[2] as OpenAI.Responses.ResponseInputItem.FunctionCallOutput;
        expect(synthetic.type).toBe('function_call_output');
        expect(synthetic.call_id).toBe('fc_1');
        expect(synthetic.output).toContain('Tool interrupted');
        expect(synthetic.output).toContain('ask_user');

        expect(result[3]).toEqual({ role: 'user', content: 'Actually, stop that' });
    });

    test('handles multiple orphaned function_calls', () => {
        const items: ResponseInputItem[] = [
            { type: 'function_call', call_id: 'fc_1', name: 'search', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call', call_id: 'fc_2', name: 'fetch', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call', call_id: 'fc_3', name: 'process', arguments: '{}' } as ResponseInputItem,
            { role: 'user', content: 'Stop!' },
        ];

        const result = fixOrphanedToolUseOpenAI(items);

        // 3 function_calls + 3 synthetic outputs + 1 user message
        expect(result).toHaveLength(7);

        expect((result[0] as any).type).toBe('function_call');
        expect((result[1] as any).type).toBe('function_call');
        expect((result[2] as any).type).toBe('function_call');

        // 3 synthetic outputs
        expect((result[3] as any).type).toBe('function_call_output');
        expect((result[3] as any).call_id).toBe('fc_1');
        expect((result[4] as any).type).toBe('function_call_output');
        expect((result[4] as any).call_id).toBe('fc_2');
        expect((result[5] as any).type).toBe('function_call_output');
        expect((result[5] as any).call_id).toBe('fc_3');

        expect((result[6] as any).role).toBe('user');
    });

    test('handles partial outputs - only injects for missing ones', () => {
        const items: ResponseInputItem[] = [
            { type: 'function_call', call_id: 'fc_1', name: 'search', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call', call_id: 'fc_2', name: 'fetch', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc_1', output: 'Search completed' } as ResponseInputItem,
            { role: 'user', content: 'Continue' },
        ];

        const result = fixOrphanedToolUseOpenAI(items);

        // fc_1 has output, fc_2 is orphaned
        expect(result).toHaveLength(5);

        expect((result[0] as any).type).toBe('function_call');
        expect((result[1] as any).type).toBe('function_call');
        expect((result[2] as any).type).toBe('function_call_output');
        expect((result[2] as any).call_id).toBe('fc_1');

        // Synthetic output for fc_2 injected before user message
        expect((result[3] as any).type).toBe('function_call_output');
        expect((result[3] as any).call_id).toBe('fc_2');
        expect((result[3] as any).output).toContain('Tool interrupted');

        expect((result[4] as any).role).toBe('user');
    });

    test('handles trailing orphan at end of conversation', () => {
        const items: ResponseInputItem[] = [
            { role: 'user', content: 'Do something' },
            { role: 'assistant', content: 'I will use a tool.' },
            { type: 'function_call', call_id: 'fc_1', name: 'ask_user', arguments: '{"question":"What?"}' } as ResponseInputItem,
        ];

        const result = fixOrphanedToolUseOpenAI(items);

        // Trailing orphan should get a synthetic output appended
        expect(result).toHaveLength(4);
        expect((result[2] as any).type).toBe('function_call');
        expect((result[3] as any).type).toBe('function_call_output');
        expect((result[3] as any).call_id).toBe('fc_1');
    });

    test('real-world scenario: user stops agent mid-execution (ask_user)', () => {
        const items: ResponseInputItem[] = [
            // Initial conversation
            { role: 'user', content: [{ type: 'input_text', text: 'hey GPT, test workstreams' }] },
            { role: 'assistant', content: 'Workstream test successful.' },
            // User sends new message
            { role: 'user', content: [{ type: 'input_text', text: 'go berserk' }] },
            // Model calls ask_user to clarify
            { type: 'function_call', call_id: 'fc_ask_user', name: 'ask_user', arguments: '{"questions":["What do you mean?"]}' } as ResponseInputItem,
            // User sends another message (agent was stopped, no function_call_output)
            { role: 'user', content: [{ type: 'input_text', text: 'launch 10 workstreams' }] },
        ];

        const result = fixOrphanedToolUseOpenAI(items);

        expect(result).toHaveLength(6); // original 5 + 1 synthetic output

        // Synthetic function_call_output should appear before the last user message
        const synthetic = result[4] as OpenAI.Responses.ResponseInputItem.FunctionCallOutput;
        expect(synthetic.type).toBe('function_call_output');
        expect(synthetic.call_id).toBe('fc_ask_user');
        expect(synthetic.output).toContain('Tool interrupted');
        expect(synthetic.output).toContain('ask_user');

        // Last user message is preserved
        expect((result[5] as any).role).toBe('user');
    });

    test('does not inject when function_call_output exists later in the array', () => {
        const items: ResponseInputItem[] = [
            { role: 'user', content: 'Search for something' },
            { type: 'function_call', call_id: 'fc_1', name: 'search', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc_1', output: 'Results found' } as ResponseInputItem,
            { role: 'assistant', content: 'Found results.' },
            { type: 'function_call', call_id: 'fc_2', name: 'analyze', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc_2', output: 'Analysis complete' } as ResponseInputItem,
        ];

        const result = fixOrphanedToolUseOpenAI(items);

        // No changes - all function_calls have matching outputs
        expect(result).toEqual(items);
    });
});
