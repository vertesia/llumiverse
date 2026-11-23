import { type ExecutionOptions, Providers, type ToolDefinition } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from './index.js';
import {
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsPrompt,
    OpenAIChatCompletionsProtocol,
    type OpenAIChatCompletionsResponse,
} from './openai_chat_completions.js';

const tools: ToolDefinition[] = [
    { name: 'web_search', input_schema: { type: 'object' } },
    { name: 'mcp.github.search', input_schema: { type: 'object' } },
];
const EMITTED = ['', 'functions.x', 'mcp.github.search', 'web_search'];
const SENT = ['_', 'functions_x', 'mcp.github.search', 'web_search'];

describe('OpenAI Responses replayed tool names', () => {
    class TestResponsesDriver extends OpenAIResponsesDriverBase {
        provider: OpenAIResponsesDriverBase['provider'] = Providers.openai;
        service: OpenAI;

        constructor(create: (request: unknown) => Promise<unknown>) {
            super({});
            this.service = { responses: { create } } as unknown as OpenAI;
        }
    }

    const history = (): OpenAI.Responses.ResponseInputItem[] => [
        { type: 'message', role: 'user', content: 'go' },
        ...EMITTED.flatMap((name, i): OpenAI.Responses.ResponseInputItem[] => [
            { type: 'function_call', call_id: `c${i}`, name, arguments: '{}' },
            { type: 'function_call_output', call_id: `c${i}`, output: 'Tool not found' },
        ]),
    ];

    it.each([false, true])('sends safe names when streaming=%s, without changing the conversation', async (stream) => {
        const create = vi.fn(async (_request: unknown) =>
            stream
                ? (async function* () {})()
                : {
                      id: 'r',
                      status: 'completed',
                      output: [
                          {
                              id: 'm',
                              type: 'message',
                              role: 'assistant',
                              status: 'completed',
                              content: [{ type: 'output_text', text: 'ok', annotations: [] }],
                          },
                      ],
                      usage: { input_tokens: 1, output_tokens: 1 },
                  },
        );
        const driver = new TestResponsesDriver(create);
        const conversation = history();
        const options: ExecutionOptions = { model: 'gpt-5', tools, conversation };
        const prompt: OpenAI.Responses.ResponseInputItem[] = [{ type: 'message', role: 'user', content: 'again' }];

        if (stream) {
            for await (const _chunk of await driver.requestTextCompletionStream(prompt, options)) {
                // drain
            }
        } else {
            await driver.requestTextCompletion(prompt, options);
        }

        const input = (create.mock.calls[0][0] as { input: OpenAI.Responses.ResponseInputItem[] }).input;
        const names = input.flatMap((item) => (item.type === 'function_call' ? [item.name] : []));
        expect(names).toEqual(SENT);
        expect(conversation).toEqual(history());
    });
});

describe('OpenAI Chat Completions replayed tool names', () => {
    class TestProtocol extends OpenAIChatCompletionsProtocol<undefined> {
        payloads: OpenAIChatCompletionsPayload[] = [];

        constructor() {
            super({ modelName: 'test/model' });
        }

        protected async postChatCompletion(
            _driver: undefined,
            payload: OpenAIChatCompletionsPayload,
        ): Promise<OpenAIChatCompletionsResponse> {
            this.payloads.push(payload);
            return {
                id: 'chatcmpl',
                object: 'chat.completion',
                created: 1,
                model: 'test/model',
                choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
            } as unknown as OpenAIChatCompletionsResponse;
        }

        protected async postChatCompletionStream(): Promise<ReadableStream> {
            throw new Error('not used');
        }
    }

    const history = (): OpenAIChatCompletionsPrompt => ({
        _is_openai_chat_completions: true,
        messages: [
            { role: 'user', content: 'go' },
            {
                role: 'assistant',
                content: null,
                tool_calls: EMITTED.map((name, i) => ({
                    id: `c${i}`,
                    type: 'function' as const,
                    function: { name, arguments: '{}' },
                })),
            },
            ...EMITTED.map((_name, i) => ({ role: 'tool', tool_call_id: `c${i}`, content: 'Tool not found' })),
        ],
    });

    it('sends safe names, without changing the conversation', async () => {
        const protocol = new TestProtocol();
        const conversation = history();

        await protocol.requestTextCompletion(
            undefined,
            { _is_openai_chat_completions: true, messages: [{ role: 'user', content: 'again' }] },
            { model: 'test/model', model_options: { _option_id: 'text-fallback' }, tools, conversation },
        );

        const sent = protocol.payloads[0].messages.find((message) => message.tool_calls?.length);
        expect(sent?.tool_calls?.map((call) => call.function.name)).toEqual(SENT);
        expect(conversation).toEqual(history());
    });
});
