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

import type { MessageParam } from '@anthropic-ai/sdk/resources/index.js';
import type { Message } from '@aws-sdk/client-bedrock-runtime';
import type OpenAI from 'openai';
import { describe, expect, test } from 'vitest';
import {
    fixOrphanedToolResults as fixOrphanedToolResultsBedrock,
    fixOrphanedToolUse as fixOrphanedToolUseBedrock,
} from '../src/bedrock/index.js';
import {
    fixOrphanedToolResults as fixOrphanedToolResultsOpenAI,
    fixOrphanedToolUse as fixOrphanedToolUseOpenAI,
} from '../src/openai/index.js';
import {
    fixOrphanedToolResults,
    fixOrphanedToolUse as fixOrphanedToolUseClaude,
} from '../src/shared/claude-messages.js';
import type { Tree } from './__helpers__/test-utils.js';

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
                ],
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'tool_1', content: 'Search result' }],
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
                ],
            },
            {
                role: 'user',
                content: [{ type: 'text', text: 'Actually, stop that and do something else' }],
            },
        ];

        const result = fixOrphanedToolUseClaude(messages);

        expect(result).toHaveLength(2);
        expect(result[0].role).toBe('assistant');
        expect(result[1].role).toBe('user');

        // The user message should have synthetic tool_result prepended
        const userContent = result[1].content as Array<{
            type: string;
            tool_use_id?: string;
            content?: string;
            text?: string;
        }>;
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
                ],
            },
            {
                role: 'user',
                content: [{ type: 'text', text: 'Stop!' }],
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
                ],
            },
            {
                role: 'user',
                content: [
                    { type: 'tool_result', tool_use_id: 'tool_1', content: 'Search completed' },
                    { type: 'text', text: 'Continue with this result' },
                ],
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
                content: [{ type: 'tool_use', id: 'tool_1', name: 'search', input: {} }],
            },
            {
                role: 'user',
                content: 'Stop and do something else',
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
                content: [{ type: 'text', text: 'I will search for documents.' }],
            },
            { role: 'user', content: [{ type: 'text', text: 'Yes, proceed' }] },
            // Assistant starts tool execution but user stops it
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'Searching...' },
                    {
                        type: 'tool_use',
                        id: 'toolu_search',
                        name: 'search_documents',
                        input: { query: 'important docs' },
                    },
                    { type: 'tool_use', id: 'toolu_analyze', name: 'analyze_results', input: {} },
                ],
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

describe('fixOrphanedToolResults - Claude', () => {
    test('returns empty array for empty input', () => {
        expect(fixOrphanedToolResults([])).toEqual([]);
    });

    test('keeps tool_result that has a matching tool_use in the previous message', () => {
        const messages: MessageParam[] = [
            { role: 'assistant', content: [{ type: 'tool_use', id: 'tool_1', name: 'search', input: {} }] },
            { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'tool_1', content: 'ok' }] },
        ];
        expect(fixOrphanedToolResults(messages)).toEqual(messages);
    });

    test('keeps all results of a parallel batch when both tool_uses are present', () => {
        const messages: MessageParam[] = [
            {
                role: 'assistant',
                content: [
                    { type: 'tool_use', id: 'tool_1', name: 'a', input: {} },
                    { type: 'tool_use', id: 'tool_2', name: 'b', input: {} },
                ],
            },
            {
                role: 'user',
                content: [
                    { type: 'tool_result', tool_use_id: 'tool_1', content: 'r1' },
                    { type: 'tool_result', tool_use_id: 'tool_2', content: 'r2' },
                ],
            },
        ];
        expect(fixOrphanedToolResults(messages)).toEqual(messages);
    });

    test('drops a tool_result whose tool_use is absent from the previous message', () => {
        // Compaction kept the result for tool_2 but dropped the assistant tool_use for it.
        const messages: MessageParam[] = [
            { role: 'assistant', content: [{ type: 'tool_use', id: 'tool_1', name: 'a', input: {} }] },
            {
                role: 'user',
                content: [
                    { type: 'tool_result', tool_use_id: 'tool_1', content: 'r1' },
                    { type: 'tool_result', tool_use_id: 'tool_2', content: 'orphan' },
                ],
            },
        ];

        const result = fixOrphanedToolResults(messages);
        const userContent = result[1].content as Array<{ type: string; tool_use_id?: string }>;
        expect(userContent).toHaveLength(1);
        expect(userContent[0].tool_use_id).toBe('tool_1');
    });

    test('drops a tool_result whose previous message is not an assistant tool_use turn', () => {
        // e.g. a compaction summary user message precedes a surviving tool_result.
        const messages: MessageParam[] = [
            { role: 'user', content: [{ type: 'text', text: 'Here is the summary of prior work...' }] },
            { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'gone', content: 'orphan' }] },
        ];

        const result = fixOrphanedToolResults(messages);
        // The orphaned tool_result message is dropped entirely; the summary text remains.
        expect(result).toHaveLength(1);
        expect(result[0].role).toBe('user');
        const remaining = result[0].content as Array<{ type: string }>;
        expect(remaining[0].type).toBe('text');
    });

    test('preserves non-tool_result blocks while dropping the orphan', () => {
        const messages: MessageParam[] = [
            { role: 'assistant', content: [{ type: 'text', text: 'thinking out loud' }] },
            {
                role: 'user',
                content: [
                    { type: 'tool_result', tool_use_id: 'orphan', content: 'x' },
                    { type: 'text', text: 'continue please' },
                ],
            },
        ];

        const result = fixOrphanedToolResults(messages);
        const userContent = result[1].content as Array<{ type: string; text?: string }>;
        expect(userContent).toHaveLength(1);
        expect(userContent[0].type).toBe('text');
        expect(userContent[0].text).toBe('continue please');
    });

    test('regression: split parallel results recombined by merge are all retained', () => {
        // After mergeConsecutiveUserMessages, both results sit in one user message
        // whose previous assistant turn declared both tool_uses — none are orphans.
        const messages: MessageParam[] = [
            {
                role: 'assistant',
                content: [
                    { type: 'tool_use', id: 'get_project', name: 'get_project', input: {} },
                    { type: 'tool_use', id: 'learn_ops', name: 'learn_artifact_operations', input: {} },
                ],
            },
            {
                role: 'user',
                content: [
                    { type: 'tool_result', tool_use_id: 'get_project', content: 'p' },
                    { type: 'tool_result', tool_use_id: 'learn_ops', content: 'o' },
                ],
            },
        ];
        const result = fixOrphanedToolResults(messages);
        expect((result[1].content as unknown[]).length).toBe(2);
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
                ],
            },
            {
                role: 'user',
                content: [{ toolResult: { toolUseId: 'tool_1', content: [{ text: 'Search result' }] } }],
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
                ],
            },
            {
                role: 'user',
                content: [{ text: 'Actually, stop that and do something else' }],
            },
        ];

        const result = fixOrphanedToolUseBedrock(messages);

        expect(result).toHaveLength(2);
        expect(result[0].role).toBe('assistant');
        expect(result[1].role).toBe('user');

        // The user message should have synthetic toolResult prepended
        // biome-ignore lint/style/noNonNullAssertion: intentional non-null assertion; TS can't prove narrowing here
        const userContent = result[1].content!;
        expect(userContent).toHaveLength(2);
        expect(userContent[0].toolResult).toBeDefined();
        expect(userContent[0].toolResult?.toolUseId).toBe('tool_1');
        expect(userContent[0].toolResult?.content?.[0].text).toContain('Tool interrupted');
        expect(userContent[0].toolResult?.content?.[0].text).toContain('search');
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
                ],
            },
            {
                role: 'user',
                content: [{ text: 'Stop!' }],
            },
        ];

        const result = fixOrphanedToolUseBedrock(messages);

        // biome-ignore lint/style/noNonNullAssertion: intentional non-null assertion; TS can't prove narrowing here
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
                ],
            },
            {
                role: 'user',
                content: [
                    { toolResult: { toolUseId: 'tool_1', content: [{ text: 'Search completed' }] } },
                    { text: 'Continue with this result' },
                ],
            },
        ];

        const result = fixOrphanedToolUseBedrock(messages);

        // biome-ignore lint/style/noNonNullAssertion: intentional non-null assertion; TS can't prove narrowing here
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
                    {
                        toolUse: {
                            toolUseId: 'toolu_search',
                            name: 'search_documents',
                            input: { query: 'important docs' },
                        },
                    },
                    { toolUse: { toolUseId: 'toolu_analyze', name: 'analyze_results', input: {} } },
                ],
            },
            // User sends new message (agent was stopped, no toolResults)
            { role: 'user', content: [{ text: 'Never mind, do something else instead' }] },
        ];

        const result = fixOrphanedToolUseBedrock(messages);

        expect(result).toHaveLength(5);

        // Check that the last user message has synthetic toolResults
        // biome-ignore lint/style/noNonNullAssertion: intentional non-null assertion; TS can't prove narrowing here
        const lastUserContent = result[4].content!;
        expect(lastUserContent).toHaveLength(3); // 2 synthetic + 1 text

        expect(lastUserContent[0].toolResult?.toolUseId).toBe('toolu_search');
        expect(lastUserContent[0].toolResult?.content?.[0].text).toContain('user stopped');

        expect(lastUserContent[1].toolResult?.toolUseId).toBe('toolu_analyze');
        expect(lastUserContent[1].toolResult?.content?.[0].text).toContain('user stopped');

        expect(lastUserContent[2].text).toBe('Never mind, do something else instead');
    });
});

describe('fixOrphanedToolResults - Bedrock', () => {
    test('returns empty array for empty input', () => {
        expect(fixOrphanedToolResultsBedrock([])).toEqual([]);
    });

    test('keeps a toolResult that has a matching toolUse in the previous message', () => {
        const messages: Message[] = [
            { role: 'assistant', content: [{ toolUse: { toolUseId: 'tool_1', name: 'search', input: {} } }] },
            { role: 'user', content: [{ toolResult: { toolUseId: 'tool_1', content: [{ text: 'ok' }] } }] },
        ];
        expect(fixOrphanedToolResultsBedrock(messages)).toEqual(messages);
    });

    test('drops a toolResult whose toolUse is absent from the previous message', () => {
        const messages: Message[] = [
            { role: 'assistant', content: [{ toolUse: { toolUseId: 'tool_1', name: 'search', input: {} } }] },
            {
                role: 'user',
                content: [
                    { toolResult: { toolUseId: 'tool_1', content: [{ text: 'r1' }] } },
                    { toolResult: { toolUseId: 'tool_2', content: [{ text: 'orphan' }] } },
                ],
            },
        ];

        const result = fixOrphanedToolResultsBedrock(messages);
        // biome-ignore lint/style/noNonNullAssertion: content is set above
        const userContent = result[1].content!;
        expect(userContent).toHaveLength(1);
        expect(userContent[0].toolResult?.toolUseId).toBe('tool_1');
    });

    test('drops a user message that becomes empty after removing orphaned toolResults', () => {
        // Compaction left a toolResult whose assistant toolUse turn was summarized away.
        const messages: Message[] = [
            { role: 'user', content: [{ text: '[summary of prior work]' }] },
            { role: 'user', content: [{ toolResult: { toolUseId: 'gone', content: [{ text: 'orphan' }] } }] },
        ];

        const result = fixOrphanedToolResultsBedrock(messages);
        expect(result).toHaveLength(1);
        // biome-ignore lint/style/noNonNullAssertion: content is set above
        expect(result[0].content![0].text).toBe('[summary of prior work]');
    });

    test('preserves non-toolResult blocks while dropping the orphan', () => {
        const messages: Message[] = [
            { role: 'assistant', content: [{ text: 'thinking' }] },
            {
                role: 'user',
                content: [
                    { toolResult: { toolUseId: 'orphan', content: [{ text: 'x' }] } },
                    { text: 'continue please' },
                ],
            },
        ];

        const result = fixOrphanedToolResultsBedrock(messages);
        // biome-ignore lint/style/noNonNullAssertion: content is set above
        const userContent = result[1].content!;
        expect(userContent).toHaveLength(1);
        expect(userContent[0].text).toBe('continue please');
    });
});

describe('fixOrphanedToolUse - OpenAI', () => {
    test('returns empty array for empty input', () => {
        const result = fixOrphanedToolUseOpenAI([]);
        expect(result).toEqual([]);
    });

    test('returns single item unchanged', () => {
        const items: ResponseInputItem[] = [{ role: 'user', content: 'Hello' }];
        const result = fixOrphanedToolUseOpenAI(items);
        expect(result).toEqual(items);
    });

    test('returns unchanged array when no orphaned function_call', () => {
        const items: ResponseInputItem[] = [
            { role: 'user', content: 'Search for something' },
            { role: 'assistant', content: 'I will search.' },
            {
                type: 'function_call',
                call_id: 'fc_1',
                name: 'search',
                arguments: '{"query":"test"}',
            } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc_1', output: 'Search result' } as ResponseInputItem,
            { role: 'assistant', content: 'Here are the results.' },
        ];

        const result = fixOrphanedToolUseOpenAI(items);
        expect(result).toEqual(items);
    });

    test('injects synthetic function_call_output for orphaned function_call followed by user message', () => {
        const items: ResponseInputItem[] = [
            { role: 'assistant', content: 'Using a tool...' },
            {
                type: 'function_call',
                call_id: 'fc_1',
                name: 'ask_user',
                arguments: '{"question":"What?"}',
            } as ResponseInputItem,
            { role: 'user', content: 'Actually, stop that' },
        ];

        const result = fixOrphanedToolUseOpenAI(items);

        expect(result).toHaveLength(4);
        expect(result[0]).toEqual({ role: 'assistant', content: 'Using a tool...' });
        expect(result[1]).toEqual({
            type: 'function_call',
            call_id: 'fc_1',
            name: 'ask_user',
            arguments: '{"question":"What?"}',
        });

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

        expect((result[0] as unknown as Tree).type).toBe('function_call');
        expect((result[1] as unknown as Tree).type).toBe('function_call');
        expect((result[2] as unknown as Tree).type).toBe('function_call');

        // 3 synthetic outputs
        expect((result[3] as unknown as Tree).type).toBe('function_call_output');
        expect((result[3] as unknown as Tree).call_id).toBe('fc_1');
        expect((result[4] as unknown as Tree).type).toBe('function_call_output');
        expect((result[4] as unknown as Tree).call_id).toBe('fc_2');
        expect((result[5] as unknown as Tree).type).toBe('function_call_output');
        expect((result[5] as unknown as Tree).call_id).toBe('fc_3');

        expect((result[6] as unknown as Tree).role).toBe('user');
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

        expect((result[0] as unknown as Tree).type).toBe('function_call');
        expect((result[1] as unknown as Tree).type).toBe('function_call');
        expect((result[2] as unknown as Tree).type).toBe('function_call_output');
        expect((result[2] as unknown as Tree).call_id).toBe('fc_1');

        // Synthetic output for fc_2 injected before user message
        expect((result[3] as unknown as Tree).type).toBe('function_call_output');
        expect((result[3] as unknown as Tree).call_id).toBe('fc_2');
        expect((result[3] as unknown as Tree).output).toContain('Tool interrupted');

        expect((result[4] as unknown as Tree).role).toBe('user');
    });

    test('handles trailing orphan at end of conversation', () => {
        const items: ResponseInputItem[] = [
            { role: 'user', content: 'Do something' },
            { role: 'assistant', content: 'I will use a tool.' },
            {
                type: 'function_call',
                call_id: 'fc_1',
                name: 'ask_user',
                arguments: '{"question":"What?"}',
            } as ResponseInputItem,
        ];

        const result = fixOrphanedToolUseOpenAI(items);

        // Trailing orphan should get a synthetic output appended
        expect(result).toHaveLength(4);
        expect((result[2] as unknown as Tree).type).toBe('function_call');
        expect((result[3] as unknown as Tree).type).toBe('function_call_output');
        expect((result[3] as unknown as Tree).call_id).toBe('fc_1');
    });

    test('real-world scenario: user stops agent mid-execution (ask_user)', () => {
        const items: ResponseInputItem[] = [
            // Initial conversation
            { role: 'user', content: [{ type: 'input_text', text: 'hey GPT, test workstreams' }] },
            { role: 'assistant', content: 'Workstream test successful.' },
            // User sends new message
            { role: 'user', content: [{ type: 'input_text', text: 'go berserk' }] },
            // Model calls ask_user to clarify
            {
                type: 'function_call',
                call_id: 'fc_ask_user',
                name: 'ask_user',
                arguments: '{"questions":["What do you mean?"]}',
            } as ResponseInputItem,
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
        expect((result[5] as unknown as Tree).role).toBe('user');
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

describe('fixOrphanedToolResults - OpenAI', () => {
    test('returns empty array for empty input', () => {
        expect(fixOrphanedToolResultsOpenAI([])).toEqual([]);
    });

    test('keeps a function_call_output that has a matching function_call', () => {
        const items: ResponseInputItem[] = [
            { type: 'function_call', call_id: 'fc_1', name: 'search', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc_1', output: 'ok' } as ResponseInputItem,
        ];
        expect(fixOrphanedToolResultsOpenAI(items)).toEqual(items);
    });

    test('drops a function_call_output whose call_id has no function_call', () => {
        const items: ResponseInputItem[] = [
            { type: 'function_call', call_id: 'fc_1', name: 'search', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc_1', output: 'r1' } as ResponseInputItem,
            // Orphan: its function_call was compacted away.
            { type: 'function_call_output', call_id: 'fc_gone', output: 'orphan' } as ResponseInputItem,
            { role: 'assistant', content: 'done' },
        ];

        const result = fixOrphanedToolResultsOpenAI(items);
        expect(result).toHaveLength(3);
        expect(result.some((i) => 'call_id' in i && i.call_id === 'fc_gone')).toBe(false);
        expect(result.some((i) => 'call_id' in i && i.call_id === 'fc_1')).toBe(true);
    });

    test('retains parallel outputs that all have matching calls', () => {
        const items: ResponseInputItem[] = [
            { type: 'function_call', call_id: 'fc_1', name: 'a', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call', call_id: 'fc_2', name: 'b', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc_2', output: 'r2' } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc_1', output: 'r1' } as ResponseInputItem,
        ];
        expect(fixOrphanedToolResultsOpenAI(items)).toEqual(items);
    });

    test('leaves non-function items untouched', () => {
        const items: ResponseInputItem[] = [
            { role: 'user', content: 'hello' },
            { role: 'assistant', content: 'hi' },
        ];
        expect(fixOrphanedToolResultsOpenAI(items)).toEqual(items);
    });
});
