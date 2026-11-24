import type { ConverseRequest } from '@aws-sdk/client-bedrock-runtime';
import type { ExecutionOptions, ToolDefinition } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from './index.js';

const MODEL = 'amazon.nova-pro-v1:0';
const tools: ToolDefinition[] = [
    { name: 'web_search', input_schema: { type: 'object' } },
    { name: 'mcp.github.search', input_schema: { type: 'object' } },
];

function conversation(): ConverseRequest {
    return {
        modelId: MODEL,
        messages: [
            { role: 'user', content: [{ text: 'go' }] },
            {
                role: 'assistant',
                content: [
                    { toolUse: { toolUseId: 'c1', name: '', input: {} } },
                    { toolUse: { toolUseId: 'c2', name: 'multi_tool_use.parallel', input: {} } },
                    { toolUse: { toolUseId: 'c3', name: 'mcp.github.search', input: {} } },
                    { toolUse: { toolUseId: 'c4', name: 'web_search', input: {} } },
                ],
            },
            {
                role: 'user',
                content: ['c1', 'c2', 'c3', 'c4'].map((id) => ({
                    toolResult: { toolUseId: id, content: [{ text: 'Tool not found' }] },
                })),
            },
        ],
    };
}

describe('Bedrock Converse replayed tool names', () => {
    it('sends safe names for undeclared tools and keeps declared ones, without changing the conversation', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const info = vi.fn();
        driver.logger = { debug: vi.fn(), info, warn: vi.fn(), error: vi.fn() };
        const prompt = conversation();
        const options: ExecutionOptions = { model: MODEL, tools };

        const request = driver.preparePayload(prompt, options);

        const names = request.messages?.[1].content?.map((block) => block.toolUse?.name);
        expect(names).toEqual(['_', 'multi_tool_use_parallel', 'mcp.github.search', 'web_search']);
        expect(prompt).toEqual(conversation());
        expect(info).toHaveBeenCalledWith(
            expect.objectContaining({
                tool_name_rewrites: [
                    { from: '', to: '_' },
                    { from: 'multi_tool_use.parallel', to: 'multi_tool_use_parallel' },
                ],
            }),
            expect.any(String),
        );
    });

    it('keeps the emitted names when a request without tools sends the history as text', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });

        const request = driver.preparePayload(conversation(), { model: MODEL });

        const text = JSON.stringify(request.messages);
        expect(text).toContain('[Tool call: multi_tool_use.parallel(');
        expect(text).not.toContain('multi_tool_use_parallel');
    });

    it('logs nothing when every replayed name is declared', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const info = vi.fn();
        driver.logger = { debug: vi.fn(), info, warn: vi.fn(), error: vi.fn() };

        driver.preparePayload(
            {
                modelId: MODEL,
                messages: [
                    { role: 'assistant', content: [{ toolUse: { toolUseId: 'c1', name: 'web_search', input: {} } }] },
                ],
            },
            { model: MODEL, tools },
        );

        expect(info).not.toHaveBeenCalled();
    });
});
