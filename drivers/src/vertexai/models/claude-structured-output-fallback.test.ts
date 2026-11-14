import { BadRequestError } from '@anthropic-ai/sdk/error';
import type { Message, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/index.js';
import { JSON_SCHEMA_INSTRUCTION_PREFIX } from '@llumiverse/common';
import type { CompletionChunkObject, ExecutionOptions, JSONSchema } from '@llumiverse/core';
import { PromptRole } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { claudeSdkStream } from '../../../test/__helpers__/test-utils.js';
import type { VertexAIDriver } from '../index.js';
import { ClaudeModelDefinition } from './claude.js';

const MODEL_PATH = 'locations/global/publishers/anthropic/models/claude-haiku-4-5';
const schema = {
    type: 'object',
    properties: { name: { type: 'string' } },
    required: ['name'],
    additionalProperties: false,
} as JSONSchema;

function policyDenial(): BadRequestError {
    const body = {
        error: {
            code: 400,
            message:
                'Organization Policy constraint constraints/vertexai.allowedPartnerModelFeatures violated for ' +
                '`projects/123` attempting to use a disallowed feature structured_outputs for Partner model claude-haiku-4-5.',
            status: 'FAILED_PRECONDITION',
        },
    };
    return new BadRequestError(400, body, `400 ${JSON.stringify([body])}`, new Headers());
}

function otherBadRequest(): BadRequestError {
    const body = { type: 'error', error: { type: 'invalid_request_error', message: 'max_tokens: too large' } };
    return new BadRequestError(400, body, `400 ${JSON.stringify(body)}`, new Headers());
}

const answer: Message = {
    id: 'msg_1',
    type: 'message',
    role: 'assistant',
    model: 'claude-haiku-4-5',
    content: [{ type: 'text', text: '{"name":"Ada"}', citations: null }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: 10, output_tokens: 5 },
} as unknown as Message;

const answerEvents = [
    { type: 'message_start', message: { ...answer, content: [] } },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: '{"name":"Ada"}' } },
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 5 } },
] as unknown as RawMessageStreamEvent[];

function failingStream(error: Error) {
    return {
        // biome-ignore lint/correctness/useYield: the request fails before its first event.
        async *[Symbol.asyncIterator]() {
            throw error;
        },
        async finalMessage(): Promise<Message> {
            throw error;
        },
    };
}

interface FakeVertex {
    driver: VertexAIDriver;
    payloads: Record<string, unknown>[];
    warnings: string[];
}

/** A Vertex AI driver whose Claude client fails the first `failures.length` requests with those errors. */
function fakeVertex(project: string, failures: Error[]): FakeVertex {
    const payloads: Record<string, unknown>[] = [];
    const warnings: string[] = [];
    const client = {
        messages: {
            stream: (payload: Record<string, unknown>) => {
                payloads.push(payload);
                const failure = failures.shift();
                return failure ? failingStream(failure) : claudeSdkStream(answerEvents, answer);
            },
        },
    };
    const driver = {
        provider: 'vertexai',
        options: { project, region: 'us-central1' },
        logger: {
            debug: () => {},
            info: () => {},
            error: () => {},
            warn: (_obj: unknown, message: string) => warnings.push(message),
        },
        getAnthropicClient: async () => client,
    } as unknown as VertexAIDriver;
    return { driver, payloads, warnings };
}

const options: ExecutionOptions = { model: MODEL_PATH, result_schema: schema };
const segments = [{ role: PromptRole.user, content: 'Name a mathematician.' }];

function schemaInstructionsIn(payload: Record<string, unknown>): boolean {
    return JSON.stringify(payload.system ?? []).includes('JSON Schema');
}

async function collect(stream: AsyncIterable<CompletionChunkObject>): Promise<string> {
    let text = '';
    for await (const chunk of stream) {
        for (const part of chunk.result ?? []) if (part.type === 'text') text += part.value;
    }
    return text;
}

describe('Vertex AI Claude structured outputs denied by an organization policy', () => {
    it('retries an execute once on the prompt path, with the schema instructions and no output_config', async () => {
        const definition = new ClaudeModelDefinition('claude-haiku-4-5');
        const { driver, payloads, warnings } = fakeVertex('execute-denied', [policyDenial()]);
        const prompt = await definition.createPrompt(driver, segments, options);

        const completion = await definition.requestTextCompletion(driver, prompt, options);

        expect(completion.result).toEqual([{ type: 'text', value: '{"name":"Ada"}' }]);
        expect(payloads).toHaveLength(2);
        expect(payloads[0].output_config).toEqual({ format: { type: 'json_schema', schema } });
        expect(schemaInstructionsIn(payloads[0])).toBe(false);
        expect(payloads[1].output_config).toBeUndefined();
        expect(schemaInstructionsIn(payloads[1])).toBe(true);
        expect(warnings).toHaveLength(1);
    });

    it("retries a continuation with the current schema's instructions, not an earlier turn's", async () => {
        const definition = new ClaudeModelDefinition('claude-haiku-4-5');
        const { driver, payloads } = fakeVertex('continuation-denied', [policyDenial()]);
        const previous = { ...schema, properties: { title: { type: 'string' } }, required: ['title'] };
        const prompt = await definition.createPrompt(driver, segments, options);
        prompt.messages = [
            {
                role: 'user',
                content: [{ type: 'text', text: `${JSON_SCHEMA_INSTRUCTION_PREFIX}\n${JSON.stringify(previous)}` }],
            },
            { role: 'assistant', content: [{ type: 'text', text: '{"title":"Earlier"}' }] },
            ...prompt.messages,
        ];

        await definition.requestTextCompletion(driver, prompt, options);

        expect(payloads).toHaveLength(2);
        const retrySystem = (payloads[1].system as { text: string }[]).map((block) => block.text).join('\n');
        expect(retrySystem).toContain(`${JSON_SCHEMA_INSTRUCTION_PREFIX}\n${JSON.stringify(schema)}`);
    });

    it('retries a stream once on the prompt path', async () => {
        const definition = new ClaudeModelDefinition('claude-haiku-4-5');
        const { driver, payloads } = fakeVertex('stream-denied', [policyDenial()]);
        const prompt = await definition.createPrompt(driver, segments, options);

        const stream = await definition.requestTextCompletionStream(driver, prompt, options);

        expect(await collect(stream)).toBe('{"name":"Ada"}');
        expect(await stream.finalizeConversation?.()).toBeDefined();
        expect(payloads).toHaveLength(2);
        expect(payloads[0].output_config).toBeDefined();
        expect(payloads[1].output_config).toBeUndefined();
        expect(schemaInstructionsIn(payloads[1])).toBe(true);
    });

    it('remembers the denial, so later prompts and requests skip the native attempt', async () => {
        const definition = new ClaudeModelDefinition('claude-haiku-4-5');
        const { driver, payloads } = fakeVertex('remembered', [policyDenial()]);
        await definition.requestTextCompletion(
            driver,
            await definition.createPrompt(driver, segments, options),
            options,
        );

        const prompt = await definition.createPrompt(driver, segments, options);
        await definition.requestTextCompletion(driver, prompt, options);
        await collect(await definition.requestTextCompletionStream(driver, prompt, options));

        expect(JSON.stringify(prompt.system)).toContain('JSON Schema');
        expect(payloads).toHaveLength(4);
        expect(payloads.slice(2).map((payload) => payload.output_config)).toEqual([undefined, undefined]);
    });

    it('keeps the native path for another project', async () => {
        const definition = new ClaudeModelDefinition('claude-haiku-4-5');
        const denied = fakeVertex('denied-project', [policyDenial()]);
        await definition.requestTextCompletion(denied.driver, { messages: [] }, options);

        const allowed = fakeVertex('allowed-project', []);
        const prompt = await definition.createPrompt(allowed.driver, segments, options);
        await definition.requestTextCompletion(allowed.driver, prompt, options);

        expect(allowed.payloads).toHaveLength(1);
        expect(allowed.payloads[0].output_config).toBeDefined();
    });

    it.each([
        ['execute', 'requestTextCompletion'],
        ['stream', 'requestTextCompletionStream'],
    ] as const)('lets any other 400 fail the %s', async (_label, method) => {
        const definition = new ClaudeModelDefinition('claude-haiku-4-5');
        const { driver, payloads } = fakeVertex(`other-400-${method}`, [otherBadRequest()]);
        const prompt = await definition.createPrompt(driver, segments, options);

        const run = async () => {
            const result = await definition[method](driver, prompt, options);
            if (method === 'requestTextCompletionStream') await collect(result as AsyncIterable<CompletionChunkObject>);
        };

        await expect(run()).rejects.toBeInstanceOf(BadRequestError);
        expect(payloads).toHaveLength(1);
    });
});
