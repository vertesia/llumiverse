import type { ExecutionOptions } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import {
    buildOpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsPrompt,
    OpenAIChatCompletionsProtocol,
    type OpenAIChatCompletionsProtocolOptions,
    type OpenAIChatCompletionsResponse,
    toOpenAINonStreamingPayload,
    toOpenAIStreamingPayload,
} from './openai_chat_completions.js';

class Protocol extends OpenAIChatCompletionsProtocol<undefined> {
    payload(conversation: OpenAIChatCompletionsPrompt, options: ExecutionOptions, stream: boolean) {
        return this.buildPayload(conversation, options, stream, 'openai');
    }
    protected postChatCompletion(): Promise<OpenAIChatCompletionsResponse> {
        return Promise.reject(new Error('Dry compilation must not invoke transport'));
    }
    protected postChatCompletionStream(): Promise<ReadableStream> {
        return Promise.reject(new Error('Dry compilation must not invoke transport'));
    }
}

const conversation: OpenAIChatCompletionsPrompt = {
    _is_openai_chat_completions: true,
    messages: [{ role: 'user', content: 'request' }],
};
const config: OpenAIChatCompletionsProtocolOptions = {
    modelName: 'deployment:model',
    defaultMaxTokens: 512,
    toolSchemaMode: 'openai_strict',
    extraBody: { accepted_extra: { full: true } },
};
const options: ExecutionOptions = {
    model: 'gpt-4o-2024-08-06',
    model_options: { max_tokens: 100, temperature: 0.3, tool_choice: 'required' },
    tools: [
        {
            name: 'lookup',
            description: 'Lookup source',
            input_schema: {
                type: 'object',
                properties: { query: { type: 'string' } },
                required: ['query'],
                additionalProperties: false,
            },
        },
    ],
    result_schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'] },
};

describe('shared actual native request builder', () => {
    it('matches transport including deployment model, actual tools/options/schema and SDK stream flags', () => {
        const protocol = new Protocol(config);
        for (const stream of [false, true]) {
            const transport = protocol.payload(conversation, options, stream);
            const dry = buildOpenAIChatCompletionsPayload(
                conversation,
                options,
                config,
                config.modelName ?? options.model,
                stream,
                'openai',
            );
            expect(dry).toEqual(transport);
            const actual = stream ? toOpenAIStreamingPayload(dry) : toOpenAINonStreamingPayload(dry);
            expect(actual).toMatchObject({
                model: 'deployment:model',
                max_tokens: 100,
                tool_choice: 'required',
                tools: [{ type: 'function', function: { name: 'lookup' } }],
                response_format: { type: 'json_schema' },
                stream,
            });
            expect(actual).toHaveProperty('accepted_extra', { full: true });
            if (stream) expect(actual).toHaveProperty('stream_options', { include_usage: true });
        }
    });

    it('preserves required-tool errors and prompt-only result-schema behavior', () => {
        expect(() =>
            buildOpenAIChatCompletionsPayload(
                conversation,
                { ...options, tools: [] },
                config,
                options.model,
                false,
                'openai',
            ),
        ).toThrow('required tool choice');
        const payload = buildOpenAIChatCompletionsPayload(
            conversation,
            options,
            { ...config, resultSchemaMode: 'prompt' },
            options.model,
            false,
            'openai',
        );
        expect(payload.response_format).toBeUndefined();
        expect(payload.tools).toHaveLength(1);
    });
});
