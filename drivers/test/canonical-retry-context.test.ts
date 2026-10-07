import { appendConversationRecords, createConversationDocument } from '@llumiverse/conversation';
import type { ExecutionOptions } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import {
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsPrompt,
    OpenAIChatCompletionsProtocol,
    type OpenAIChatCompletionsResponse,
} from '../src/openai/openai_chat_completions.js';

class CanonicalRetryProtocol extends OpenAIChatCompletionsProtocol<undefined> {
    readonly requests: OpenAIChatCompletionsPayload[] = [];

    constructor() {
        super({ modelName: 'test/model' });
    }

    protected async postChatCompletion(
        _driver: undefined,
        payload: OpenAIChatCompletionsPayload,
    ): Promise<OpenAIChatCompletionsResponse> {
        this.requests.push(payload);
        return {
            id: 'response-native',
            model: 'test/model',
            object: 'chat.completion',
            created: 1,
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: 'Answer' },
                    finish_reason: 'stop',
                    logprobs: null,
                },
            ],
        };
    }

    protected async postChatCompletionStream(): Promise<ReadableStream> {
        throw new Error('This fixture exercises finite canonical execution');
    }
}

const at = '2026-09-11T00:00:00.000Z';
const prompt: OpenAIChatCompletionsPrompt = {
    _is_openai_chat_completions: true,
    messages: [{ role: 'user', content: 'Question' }],
};

describe('canonical accepted request context', () => {
    it('recovers an exact request with excluded program context without executing the provider twice', async () => {
        const document = appendConversationRecords(
            createConversationDocument({ id: 'conversation', created_at: at }),
            {
                turns: [
                    {
                        id: 'program-event',
                        kind: 'program',
                        authority: 'ordinary',
                        model_visibility: 'exclude',
                        status: 'completed',
                        provenance: { type: 'received' },
                        timestamps: { recorded_at: at },
                        blocks: [
                            { id: 'program-text', type: 'text', text: 'Internal workflow progress', format: 'plain' },
                        ],
                    },
                ],
                context_entries: [{ id: 'program-context', type: 'source_turn', turn_id: 'program-event' }],
            },
            {
                expected_revision: 0,
                operation_id: 'program-operation',
                payload_fingerprint: 'program-payload',
                recorded_at: at,
            },
        ).document;
        const options: ExecutionOptions = {
            model: 'test/model',
            conversation: document,
            conversation_runtime: {
                conversation_id: document.id,
                request_id: 'request',
                attempt_id: 'attempt',
                input_operation_id: 'input',
                response_operation_id: 'response',
                recorded_at: at,
            },
        };
        const protocol = new CanonicalRetryProtocol();
        const first = await protocol.requestCanonicalTextCompletion(undefined, prompt, options);
        const retry = await protocol.requestCanonicalTextCompletion(undefined, prompt, {
            ...options,
            conversation: JSON.parse(JSON.stringify(first.conversation)),
        });

        expect(retry.accepted_output).toEqual(first.accepted_output);
        expect(protocol.requests).toHaveLength(1);
        expect(protocol.requests[0].messages).toEqual([{ role: 'user', content: 'Question' }]);
    });
});
