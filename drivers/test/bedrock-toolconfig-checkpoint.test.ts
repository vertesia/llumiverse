/**
 * Unit tests for the tool-block-to-text conversion in all drivers.
 *
 * When no tools are provided but conversation contains tool call/result blocks
 * (e.g. checkpoint summary calls), each driver converts those blocks to text
 * representations to avoid API errors while preserving tool call data.
 *
 * Tests the Bedrock driver's preparePayload directly (the only driver where
 * the conversion is testable at the payload level without making API calls).
 * VertexAI and OpenAI conversion functions are tested indirectly via the live
 * integration tests in checkpoint-tool-conversion.test.ts.
 */

import { describe, expect, test } from 'vitest';
import { BedrockDriver } from '../src/bedrock/index.js';
import type { Message, ConverseRequest } from '@aws-sdk/client-bedrock-runtime';
import type { ExecutionOptions } from '@llumiverse/core';

function createDriver() {
    return new BedrockDriver({ region: 'us-east-1' });
}

function makePrompt(messages: Message[], system?: ConverseRequest['system']): ConverseRequest {
    return { modelId: 'anthropic.claude-sonnet-4-20250514', messages, system };
}

function makeOptions(overrides: Partial<ExecutionOptions> = {}): ExecutionOptions {
    return {
        model: 'anthropic.claude-sonnet-4-20250514',
        ...overrides,
    };
}

describe('Bedrock preparePayload - tool blocks in conversation with empty tools', () => {

    test('converts toolUse/toolResult blocks to text when tools=[]', () => {
        const driver = createDriver();

        const messages: Message[] = [
            { role: 'user', content: [{ text: 'Search for documents' }] },
            {
                role: 'assistant',
                content: [
                    { text: 'Searching...' },
                    { toolUse: { toolUseId: 'tool_1', name: 'search_docs', input: { query: 'test' } } },
                ],
            },
            {
                role: 'user',
                content: [
                    { toolResult: { toolUseId: 'tool_1', content: [{ text: 'Found 3 documents' }] } },
                ],
            },
            {
                role: 'assistant',
                content: [{ text: 'I found 3 documents.' }],
            },
            {
                role: 'user',
                content: [{ text: 'Create a checkpoint summary' }],
            },
        ];

        const prompt = makePrompt(messages);
        const options = makeOptions({ tools: [] });

        const request = driver.preparePayload(prompt, options);

        // No toolConfig should be set (tool blocks were converted to text)
        expect(request.toolConfig).toBeUndefined();

        // Messages should still exist
        expect(request.messages).toBeDefined();
        expect(request.messages!.length).toBe(5);

        // Assistant message: toolUse block should be converted to text
        const assistantMsg = request.messages![1];
        expect(assistantMsg.content!.length).toBe(2);
        expect((assistantMsg.content![0] as any).text).toBe('Searching...');
        expect((assistantMsg.content![1] as any).text).toContain('[Tool call: search_docs');
        expect((assistantMsg.content![1] as any).text).toContain('query');
        expect((assistantMsg.content![1] as any).toolUse).toBeUndefined();

        // User message: toolResult block should be converted to text
        const toolResultMsg = request.messages![2];
        expect(toolResultMsg.content!.length).toBe(1);
        expect((toolResultMsg.content![0] as any).text).toContain('[Tool result:');
        expect((toolResultMsg.content![0] as any).text).toContain('Found 3 documents');
        expect((toolResultMsg.content![0] as any).toolResult).toBeUndefined();

        // Plain text messages should be unchanged
        expect((request.messages![0].content![0] as any).text).toBe('Search for documents');
        expect((request.messages![3].content![0] as any).text).toBe('I found 3 documents.');
        expect((request.messages![4].content![0] as any).text).toBe('Create a checkpoint summary');
    });

    test('preserves multiple tool names and deduplicates in conversion', () => {
        const driver = createDriver();

        const messages: Message[] = [
            { role: 'user', content: [{ text: 'Do stuff' }] },
            {
                role: 'assistant',
                content: [
                    { toolUse: { toolUseId: 'tool_1', name: 'think', input: { thought: 'hmm' } } },
                ],
            },
            {
                role: 'user',
                content: [{ toolResult: { toolUseId: 'tool_1', content: [{ text: 'ok' }] } }],
            },
            {
                role: 'assistant',
                content: [
                    { toolUse: { toolUseId: 'tool_2', name: 'fetch_document', input: { id: '123' } } },
                ],
            },
            {
                role: 'user',
                content: [{ toolResult: { toolUseId: 'tool_2', content: [{ text: 'doc content here' }] } }],
            },
            {
                role: 'user',
                content: [{ text: 'Summarize' }],
            },
        ];

        const prompt = makePrompt(messages);
        const options = makeOptions({ tools: [] });

        const request = driver.preparePayload(prompt, options);

        expect(request.toolConfig).toBeUndefined();

        // Check think tool was converted
        expect((request.messages![1].content![0] as any).text).toContain('[Tool call: think');
        expect((request.messages![2].content![0] as any).text).toContain('[Tool result:');
        expect((request.messages![2].content![0] as any).text).toContain('ok');

        // Check fetch_document tool was converted
        expect((request.messages![3].content![0] as any).text).toContain('[Tool call: fetch_document');
        expect((request.messages![4].content![0] as any).text).toContain('[Tool result:');
        expect((request.messages![4].content![0] as any).text).toContain('doc content here');
    });

    test('does NOT convert when tools=[] and conversation has no tool blocks', () => {
        const driver = createDriver();

        const messages: Message[] = [
            { role: 'user', content: [{ text: 'Hello' }] },
            { role: 'assistant', content: [{ text: 'Hi there!' }] },
            { role: 'user', content: [{ text: 'Summarize' }] },
        ];

        const prompt = makePrompt(messages);
        const options = makeOptions({ tools: [] });

        const request = driver.preparePayload(prompt, options);

        expect(request.toolConfig).toBeUndefined();
        // Messages unchanged
        expect((request.messages![0].content![0] as any).text).toBe('Hello');
        expect((request.messages![1].content![0] as any).text).toBe('Hi there!');
        expect((request.messages![2].content![0] as any).text).toBe('Summarize');
    });

    test('uses provided tools when tools array is non-empty (no conversion)', () => {
        const driver = createDriver();

        const messages: Message[] = [
            { role: 'user', content: [{ text: 'Search' }] },
            {
                role: 'assistant',
                content: [
                    { toolUse: { toolUseId: 'tool_1', name: 'old_tool', input: {} } },
                ],
            },
            {
                role: 'user',
                content: [{ toolResult: { toolUseId: 'tool_1', content: [{ text: 'result' }] } }],
            },
        ];

        const prompt = makePrompt(messages);
        const options = makeOptions({
            tools: [
                { name: 'new_tool', description: 'A new tool', input_schema: { type: 'object' } },
            ],
        });

        const request = driver.preparePayload(prompt, options);

        // Should use the explicitly provided tools
        expect(request.toolConfig).toBeDefined();
        const toolNames = request.toolConfig!.tools!.map((t: any) => t.toolSpec?.name);
        expect(toolNames).toEqual(['new_tool']);

        // Tool blocks in conversation should be preserved (not converted)
        expect((request.messages![1].content![0] as any).toolUse).toBeDefined();
        expect((request.messages![2].content![0] as any).toolResult).toBeDefined();
    });

    test('truncates large tool inputs and results in conversion', () => {
        const driver = createDriver();

        const largeInput = { data: 'x'.repeat(1000) };
        const largeResult = 'y'.repeat(1000);

        const messages: Message[] = [
            { role: 'user', content: [{ text: 'Go' }] },
            {
                role: 'assistant',
                content: [
                    { toolUse: { toolUseId: 'tool_1', name: 'big_tool', input: largeInput } },
                ],
            },
            {
                role: 'user',
                content: [{ toolResult: { toolUseId: 'tool_1', content: [{ text: largeResult }] } }],
            },
        ];

        const prompt = makePrompt(messages);
        const options = makeOptions({ tools: [] });

        const request = driver.preparePayload(prompt, options);

        // Tool call text should be truncated
        const toolCallText = (request.messages![1].content![0] as any).text as string;
        expect(toolCallText).toContain('[Tool call: big_tool');
        expect(toolCallText).toContain('...');
        expect(toolCallText.length).toBeLessThan(700);

        // Tool result text should be truncated
        const toolResultText = (request.messages![2].content![0] as any).text as string;
        expect(toolResultText).toContain('[Tool result:');
        expect(toolResultText).toContain('...');
        expect(toolResultText.length).toBeLessThan(700);
    });

    test('checkpoint scenario: many tool turns with tools=[]', () => {
        const driver = createDriver();

        const messages: Message[] = [
            { role: 'user', content: [{ text: 'Analyze the project' }] },
        ];

        const toolNames = ['think', 'plan', 'fetch_document', 'launch_workstream', 'list_workstreams'];
        let toolCounter = 0;

        for (let i = 0; i < 20; i++) {
            const name = toolNames[i % toolNames.length];
            const toolUseId = `tooluse_${++toolCounter}`;

            messages.push({
                role: 'assistant',
                content: [
                    { toolUse: { toolUseId, name, input: { data: `call ${i}` } } },
                ],
            });
            messages.push({
                role: 'user',
                content: [
                    { toolResult: { toolUseId, content: [{ text: `Result for ${name} call ${i}` }] } },
                ],
            });
        }

        messages.push({ role: 'assistant', content: [{ text: 'Analysis complete.' }] });
        messages.push({ role: 'user', content: [{ text: 'Create a checkpoint summary...' }] });

        const prompt = makePrompt(messages);
        const options = makeOptions({ tools: [] });

        const request = driver.preparePayload(prompt, options);

        // No toolConfig — tool blocks were converted to text
        expect(request.toolConfig).toBeUndefined();

        // No remaining toolUse/toolResult blocks in messages
        for (const msg of request.messages!) {
            for (const block of msg.content!) {
                expect((block as any).toolUse).toBeUndefined();
                expect((block as any).toolResult).toBeUndefined();
            }
        }

        // All tool calls preserved as text
        const allText = request.messages!.flatMap(m =>
            m.content!.map(b => (b as any).text || '')
        ).join('\n');

        expect(allText).toContain('[Tool call: think');
        expect(allText).toContain('[Tool call: plan');
        expect(allText).toContain('[Tool call: fetch_document');
        expect(allText).toContain('[Tool result:');
        expect(allText).toContain('Result for think call 0');
    });
});
