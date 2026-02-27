/**
 * Unit tests for the tool-block-to-text conversion across all drivers.
 *
 * When no tools are provided but conversation contains tool call/result blocks
 * (e.g. checkpoint summary calls), each driver converts those blocks to text
 * representations to avoid API errors while preserving tool call data.
 *
 * Covers: Bedrock, VertexAI Claude, VertexAI Gemini, OpenAI
 */

import { describe, expect, test } from 'vitest';

// Bedrock
import { BedrockDriver, convertToolBlocksToText, messagesContainToolBlocks } from '../src/bedrock/index.js';
import type { Message, ConverseRequest } from '@aws-sdk/client-bedrock-runtime';
import type { ExecutionOptions } from '@llumiverse/core';

// VertexAI Claude
import { claudeMessagesContainToolBlocks, convertClaudeToolBlocksToText } from '../src/vertexai/models/claude.js';
import type { MessageParam } from '@anthropic-ai/sdk/resources/index.js';

// VertexAI Gemini
import { convertGeminiFunctionPartsToText } from '../src/vertexai/models/gemini.js';
import type { Content } from '@google/genai';

// OpenAI
import { convertOpenAIFunctionItemsToText } from '../src/openai/index.js';
import type OpenAI from 'openai';
type ResponseInputItem = OpenAI.Responses.ResponseInputItem;

// ─── Bedrock ──────────────────────────────────────────────────────────────────

describe('Bedrock - convertToolBlocksToText', () => {

    test('detects tool blocks in messages', () => {
        const withTools: Message[] = [
            { role: 'user', content: [{ text: 'hi' }] },
            { role: 'assistant', content: [{ toolUse: { toolUseId: 't1', name: 'think', input: {} } }] },
        ];
        const withoutTools: Message[] = [
            { role: 'user', content: [{ text: 'hi' }] },
            { role: 'assistant', content: [{ text: 'hello' }] },
        ];
        expect(messagesContainToolBlocks(withTools)).toBe(true);
        expect(messagesContainToolBlocks(withoutTools)).toBe(false);
        expect(messagesContainToolBlocks([])).toBe(false);
    });

    test('converts toolUse to text and preserves plain blocks', () => {
        const messages: Message[] = [
            { role: 'user', content: [{ text: 'Search' }] },
            {
                role: 'assistant',
                content: [
                    { text: 'Searching...' },
                    { toolUse: { toolUseId: 't1', name: 'search_docs', input: { query: 'test' } } },
                ],
            },
            {
                role: 'user',
                content: [
                    { toolResult: { toolUseId: 't1', content: [{ text: 'Found 3 docs' }] } },
                ],
            },
        ];

        const result = convertToolBlocksToText(messages);

        // Plain message unchanged
        expect((result[0].content![0] as any).text).toBe('Search');
        // Text block preserved, toolUse converted
        expect((result[1].content![0] as any).text).toBe('Searching...');
        expect((result[1].content![1] as any).text).toContain('[Tool call: search_docs');
        expect((result[1].content![1] as any).text).toContain('query');
        expect((result[1].content![1] as any).toolUse).toBeUndefined();
        // toolResult converted
        expect((result[2].content![0] as any).text).toContain('[Tool result:');
        expect((result[2].content![0] as any).text).toContain('Found 3 docs');
        expect((result[2].content![0] as any).toolResult).toBeUndefined();
    });

    test('truncates large inputs and results', () => {
        const messages: Message[] = [
            { role: 'assistant', content: [{ toolUse: { toolUseId: 't1', name: 'big', input: { data: 'x'.repeat(1000) } } }] },
            { role: 'user', content: [{ toolResult: { toolUseId: 't1', content: [{ text: 'y'.repeat(1000) }] } }] },
        ];
        const result = convertToolBlocksToText(messages);
        expect((result[0].content![0] as any).text.length).toBeLessThan(700);
        expect((result[0].content![0] as any).text).toContain('...');
        expect((result[1].content![0] as any).text.length).toBeLessThan(700);
        expect((result[1].content![0] as any).text).toContain('...');
    });

    test('preparePayload converts when tools=[] but preserves when tools provided', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const messages: Message[] = [
            { role: 'user', content: [{ text: 'Go' }] },
            { role: 'assistant', content: [{ toolUse: { toolUseId: 't1', name: 'think', input: {} } }] },
            { role: 'user', content: [{ toolResult: { toolUseId: 't1', content: [{ text: 'ok' }] } }] },
            { role: 'user', content: [{ text: 'Summarize' }] },
        ];
        const prompt: ConverseRequest = { modelId: 'anthropic.claude-sonnet-4-20250514', messages };

        // tools=[] → conversion, no toolConfig
        const emptyTools = driver.preparePayload(prompt, { model: 'anthropic.claude-sonnet-4-20250514', tools: [] } as ExecutionOptions);
        expect(emptyTools.toolConfig).toBeUndefined();
        expect((emptyTools.messages![1].content![0] as any).toolUse).toBeUndefined();
        expect((emptyTools.messages![1].content![0] as any).text).toContain('[Tool call: think');

        // tools provided → no conversion, toolConfig set
        const withTools = driver.preparePayload(
            { modelId: 'anthropic.claude-sonnet-4-20250514', messages: [...messages] },
            { model: 'anthropic.claude-sonnet-4-20250514', tools: [{ name: 'x', description: 'x', input_schema: { type: 'object' } }] } as ExecutionOptions,
        );
        expect(withTools.toolConfig).toBeDefined();
        expect((withTools.messages![1].content![0] as any).toolUse).toBeDefined();
    });

    test('no conversion when conversation has no tool blocks', () => {
        const messages: Message[] = [
            { role: 'user', content: [{ text: 'Hello' }] },
            { role: 'assistant', content: [{ text: 'Hi' }] },
        ];
        const result = convertToolBlocksToText(messages);
        expect(result).toEqual(messages);
    });
});

// ─── VertexAI Claude ──────────────────────────────────────────────────────────

describe('VertexAI Claude - convertClaudeToolBlocksToText', () => {

    test('detects tool blocks in messages', () => {
        const withTools: MessageParam[] = [
            { role: 'user', content: [{ type: 'text', text: 'hi' }] },
            { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'think', input: {} }] },
        ];
        const withoutTools: MessageParam[] = [
            { role: 'user', content: [{ type: 'text', text: 'hi' }] },
            { role: 'assistant', content: [{ type: 'text', text: 'hello' }] },
        ];
        expect(claudeMessagesContainToolBlocks(withTools)).toBe(true);
        expect(claudeMessagesContainToolBlocks(withoutTools)).toBe(false);
        expect(claudeMessagesContainToolBlocks([])).toBe(false);
    });

    test('converts tool_use and tool_result to text', () => {
        const messages: MessageParam[] = [
            { role: 'user', content: [{ type: 'text', text: 'Search' }] },
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'Searching...' },
                    { type: 'tool_use', id: 't1', name: 'search_docs', input: { query: 'test' } },
                ],
            },
            {
                role: 'user',
                content: [
                    { type: 'tool_result', tool_use_id: 't1', content: 'Found 3 docs' },
                ],
            },
        ];

        const result = convertClaudeToolBlocksToText(messages);

        // Plain text unchanged
        const userContent = result[0].content as Array<{ type: string; text: string }>;
        expect(userContent[0].text).toBe('Search');

        // tool_use converted to text
        const assistantContent = result[1].content as Array<{ type: string; text?: string }>;
        expect(assistantContent[0].text).toBe('Searching...');
        expect(assistantContent[1].type).toBe('text');
        expect(assistantContent[1].text).toContain('[Tool call: search_docs');
        expect(assistantContent[1].text).toContain('query');

        // tool_result converted to text
        const resultContent = result[2].content as Array<{ type: string; text?: string }>;
        expect(resultContent[0].type).toBe('text');
        expect(resultContent[0].text).toContain('[Tool result:');
        expect(resultContent[0].text).toContain('Found 3 docs');
    });

    test('handles tool_result with array content', () => {
        const messages: MessageParam[] = [
            { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'fetch', input: {} }] },
            {
                role: 'user',
                content: [
                    {
                        type: 'tool_result',
                        tool_use_id: 't1',
                        content: [{ type: 'text', text: 'line 1' }, { type: 'text', text: 'line 2' }],
                    },
                ],
            },
        ];
        const result = convertClaudeToolBlocksToText(messages);
        const resultContent = result[1].content as Array<{ type: string; text?: string }>;
        expect(resultContent[0].text).toContain('line 1');
        expect(resultContent[0].text).toContain('line 2');
    });

    test('truncates large inputs and results', () => {
        const messages: MessageParam[] = [
            { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'big', input: { data: 'x'.repeat(1000) } }] },
            { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: 'y'.repeat(1000) }] },
        ];
        const result = convertClaudeToolBlocksToText(messages);
        const callText = (result[0].content as Array<{ text: string }>)[0].text;
        const resultText = (result[1].content as Array<{ text: string }>)[0].text;
        expect(callText.length).toBeLessThan(700);
        expect(callText).toContain('...');
        expect(resultText.length).toBeLessThan(700);
        expect(resultText).toContain('...');
    });

    test('no conversion when no tool blocks', () => {
        const messages: MessageParam[] = [
            { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
            { role: 'assistant', content: [{ type: 'text', text: 'Hi' }] },
        ];
        const result = convertClaudeToolBlocksToText(messages);
        expect(result).toEqual(messages);
    });
});

// ─── VertexAI Gemini ──────────────────────────────────────────────────────────

describe('VertexAI Gemini - convertGeminiFunctionPartsToText', () => {

    test('converts functionCall and functionResponse to text parts', () => {
        const contents: Content[] = [
            { role: 'user', parts: [{ text: 'What is the weather?' }] },
            {
                role: 'model',
                parts: [
                    { text: 'Let me check.' },
                    { functionCall: { name: 'get_weather', args: { location: 'Paris' } } },
                ],
            },
            {
                role: 'user',
                parts: [
                    { functionResponse: { name: 'get_weather', response: { temperature: 15 } } },
                ],
            },
        ];

        const result = convertGeminiFunctionPartsToText(contents);

        // Plain text unchanged
        expect(result[0].parts![0].text).toBe('What is the weather?');

        // functionCall converted, text preserved
        expect(result[1].parts![0].text).toBe('Let me check.');
        expect(result[1].parts![1].text).toContain('[Tool call: get_weather');
        expect(result[1].parts![1].text).toContain('Paris');
        expect(result[1].parts![1].functionCall).toBeUndefined();

        // functionResponse converted
        expect(result[2].parts![0].text).toContain('[Tool result for get_weather:');
        expect(result[2].parts![0].text).toContain('temperature');
        expect(result[2].parts![0].functionResponse).toBeUndefined();
    });

    test('truncates large args and responses', () => {
        const contents: Content[] = [
            {
                role: 'model',
                parts: [{ functionCall: { name: 'big', args: { data: 'x'.repeat(1000) } } }],
            },
            {
                role: 'user',
                parts: [{ functionResponse: { name: 'big', response: { data: 'y'.repeat(1000) } } }],
            },
        ];
        const result = convertGeminiFunctionPartsToText(contents);
        expect(result[0].parts![0].text!.length).toBeLessThan(700);
        expect(result[0].parts![0].text).toContain('...');
        expect(result[1].parts![0].text!.length).toBeLessThan(700);
        expect(result[1].parts![0].text).toContain('...');
    });

    test('no conversion when no function parts', () => {
        const contents: Content[] = [
            { role: 'user', parts: [{ text: 'Hello' }] },
            { role: 'model', parts: [{ text: 'Hi' }] },
        ];
        const result = convertGeminiFunctionPartsToText(contents);
        expect(result).toEqual(contents);
    });
});

// ─── OpenAI ───────────────────────────────────────────────────────────────────

describe('OpenAI - convertOpenAIFunctionItemsToText', () => {

    test('converts function_call and function_call_output to text messages', () => {
        const items: ResponseInputItem[] = [
            { role: 'user', content: 'What is the weather?' },
            { role: 'assistant', content: 'Let me check.' },
            { type: 'function_call', call_id: 'fc1', name: 'get_weather', arguments: '{"location":"Paris"}' } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc1', output: '15 degrees' } as ResponseInputItem,
            { role: 'assistant', content: 'It is 15 degrees.' },
        ];

        const result = convertOpenAIFunctionItemsToText(items);

        // Plain messages unchanged
        expect(result[0]).toEqual(items[0]);
        expect(result[1]).toEqual(items[1]);
        expect(result[4]).toEqual(items[4]);

        // function_call converted to assistant text
        const callItem = result[2] as any;
        expect(callItem.role).toBe('assistant');
        expect(callItem.content).toContain('[Tool call: get_weather');
        expect(callItem.content).toContain('Paris');
        expect(callItem.type).toBeUndefined();

        // function_call_output converted to user text
        const outputItem = result[3] as any;
        expect(outputItem.role).toBe('user');
        expect(outputItem.content).toContain('[Tool result:');
        expect(outputItem.content).toContain('15 degrees');
        expect(outputItem.type).toBeUndefined();
    });

    test('truncates large arguments and outputs', () => {
        const items: ResponseInputItem[] = [
            { type: 'function_call', call_id: 'fc1', name: 'big', arguments: 'x'.repeat(1000) } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc1', output: 'y'.repeat(1000) } as ResponseInputItem,
        ];
        const result = convertOpenAIFunctionItemsToText(items);
        expect((result[0] as any).content.length).toBeLessThan(700);
        expect((result[0] as any).content).toContain('...');
        expect((result[1] as any).content.length).toBeLessThan(700);
        expect((result[1] as any).content).toContain('...');
    });

    test('no conversion when no function items', () => {
        const items: ResponseInputItem[] = [
            { role: 'user', content: 'Hello' },
            { role: 'assistant', content: 'Hi' },
        ];
        const result = convertOpenAIFunctionItemsToText(items);
        expect(result).toEqual(items);
    });

    test('handles multiple function_call items', () => {
        const items: ResponseInputItem[] = [
            { type: 'function_call', call_id: 'fc1', name: 'search', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call', call_id: 'fc2', name: 'fetch', arguments: '{}' } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc1', output: 'results' } as ResponseInputItem,
            { type: 'function_call_output', call_id: 'fc2', output: 'data' } as ResponseInputItem,
            { role: 'user', content: 'Summarize' },
        ];
        const result = convertOpenAIFunctionItemsToText(items);
        expect((result[0] as any).content).toContain('[Tool call: search');
        expect((result[1] as any).content).toContain('[Tool call: fetch');
        expect((result[2] as any).content).toContain('[Tool result:');
        expect((result[3] as any).content).toContain('[Tool result:');
        expect(result[4]).toEqual(items[4]); // user message unchanged
    });
});
