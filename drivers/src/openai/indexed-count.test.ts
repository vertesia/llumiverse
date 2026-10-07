import { fingerprintJson, type ModelTarget } from '@llumiverse/conversation';
import { describe, expect, it, vi } from 'vitest';
import {
    countOpenAIIndexedRequest,
    OPENAI_INDEXED_COUNT_PROFILE,
    OPENAI_INDEXED_COUNT_VERSION,
    openAIIndexedCountBody,
} from './indexed-count.js';
import { OpenAIDriver } from './openai.js';
import { OPENAI_RESPONSES_ADAPTER_VERSION } from './openai-responses-conversation-adapter.js';

const target: ModelTarget = {
    provider: 'openai',
    protocol: 'openai.responses',
    model: 'gpt-5.4',
    adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION,
};
function request() {
    return {
        model: target.model,
        input: [
            {
                role: 'user',
                content: [
                    { type: 'input_text', text: 'Preserve the selected source.' },
                    { type: 'input_image', image_url: 'data:image/png;base64,iVBORw0KGgo=', detail: 'auto' },
                    { type: 'input_file', file_data: 'data:application/pdf;base64,JVBERi0=', filename: 'source.pdf' },
                ],
            },
            { type: 'function_call', call_id: 'call:1', name: 'read_source', arguments: '{}' },
            { type: 'function_call_output', call_id: 'call:1', output: 'Retained original output' },
        ],
        tools: [{ type: 'function', name: 'read_source', parameters: { type: 'object', properties: {} } }],
        tool_choice: 'auto',
        parallel_tool_calls: true,
        reasoning: { effort: 'low' },
        text: { verbosity: 'low' },
        stream: false,
        max_output_tokens: 2048,
        store: false,
        include: ['reasoning.encrypted_content'],
        prompt_cache_key: 'cache:owned',
        metadata: { fixture: 'retained' },
    };
}

function countedDriver(value: unknown = { object: 'response.input_tokens', input_tokens: 123 }) {
    const driver = new OpenAIDriver({ apiKey: 'never-network' });
    const transport = vi.fn<typeof fetch>(
        async () =>
            new Response(JSON.stringify(value), {
                headers: { 'content-type': 'application/json' },
            }),
    );
    driver.service = driver.service.withOptions({ fetch: transport, maxRetries: 0 });
    return { driver, transport };
}

describe('owned OpenAI Responses provider preflight count', () => {
    it('projects full input, tools and input controls while retaining exact native-body identity', async () => {
        const { driver, transport } = countedDriver();
        const native = request();
        const original = structuredClone(native);
        const counted = await driver.countIndexedNativeRequest(native, target);
        expect(transport).toHaveBeenCalledOnce();
        const transportCall = transport.mock.calls[0];
        expect(JSON.parse(String(transportCall?.[1]?.body))).toEqual({
            model: native.model,
            input: native.input,
            tools: native.tools,
            tool_choice: native.tool_choice,
            parallel_tool_calls: native.parallel_tool_calls,
            reasoning: native.reasoning,
            text: native.text,
        });
        expect(String(transportCall?.[0])).toBe('https://api.openai.com/v1/responses/input_tokens');
        expect(transportCall?.[1]?.method).toBe('POST');
        expect(transportCall?.[1]?.signal).toBeInstanceOf(AbortSignal);
        expect(new Headers(transportCall?.[1]?.headers).get('authorization')).toBe('Bearer never-network');
        expect(counted).toEqual({
            input_tokens: 123,
            profile: OPENAI_INDEXED_COUNT_PROFILE,
            tokenizer_version: OPENAI_INDEXED_COUNT_VERSION,
            request_fingerprint: await fingerprintJson(original),
        });
        expect(native).toEqual(original);
    });

    it.each([
        { previous_response_id: 'hidden:response' },
        { conversation: 'hidden:conversation' },
        { unknown_input_control: true },
        { truncation: 'auto' },
        { max_output_tokens: 'invalid' },
        { max_output_tokens: -1 },
        { max_output_tokens: 1.5 },
        { tools: [{ type: 'image_generation' }] },
    ])('refuses unregistered or hidden-history input before provider I/O: %j', async (extra) => {
        const { driver, transport } = countedDriver();
        await expect(driver.countIndexedNativeRequest({ ...request(), ...extra }, target)).rejects.toThrow();
        expect(transport).not.toHaveBeenCalled();
    });

    it.each([
        { ...target, provider: 'openai_compatible' },
        { ...target, protocol: 'openai.chat.completions' },
        { ...target, model: 'foreign:model' },
        { ...target, adapter_version: 'foreign:adapter' },
    ])('rejects a substituted target before provider I/O: %j', async (foreign) => {
        const { driver, transport } = countedDriver();
        await expect(driver.countIndexedNativeRequest(request(), foreign)).rejects.toThrow();
        expect(transport).not.toHaveBeenCalled();
    });

    it('owns mutable request bytes before the first asynchronous provider operation', async () => {
        const { driver, transport } = countedDriver();
        const native = request();
        const original = structuredClone(native);
        const counting = driver.countIndexedNativeRequest(native, target);
        native.input = [];
        const counted = await counting;
        expect(transport).toHaveBeenCalledOnce();
        expect(JSON.parse(String(transport.mock.calls[0]?.[1]?.body)).input).toEqual(original.input);
        expect(counted.request_fingerprint).toBe(await fingerprintJson(original));
    });

    it.each([
        { object: 'response.input_tokens', input_tokens: -1 },
        { object: 'response.input_tokens', input_tokens: 1.5 },
        { object: 'response.input_tokens', input_tokens: Number.MAX_SAFE_INTEGER + 1 },
        { object: 'foreign:profile', input_tokens: 123 },
        { object: 'response.input_tokens', input_tokens: 123, unexpected: true },
    ])('rejects malformed exact provider responses: %j', async (result) => {
        const { driver } = countedDriver(result);
        await expect(driver.countIndexedNativeRequest(request(), target)).rejects.toThrow();
    });

    it('bounds response bytes before JSON parsing and preserves actual SDK provider refusal', async () => {
        const { driver, transport } = countedDriver({
            object: 'response.input_tokens',
            input_tokens: 123,
            padding: 'x'.repeat(4096),
        });
        await expect(driver.countIndexedNativeRequest(request(), target)).rejects.toThrow('bounded JSON');
        transport.mockResolvedValueOnce(
            new Response(JSON.stringify({ error: { message: 'Actual provider count refusal' } }), { status: 400 }),
        );
        await expect(driver.countIndexedNativeRequest(request(), target)).rejects.toThrow(
            'Actual provider count refusal',
        );
    });

    it('bounds chunk count and cancels an unfinished response stream', async () => {
        const { driver, transport } = countedDriver();
        const cancelled = vi.fn();
        transport.mockResolvedValueOnce(
            new Response(
                new ReadableStream<Uint8Array>({
                    pull(controller) {
                        controller.enqueue(new Uint8Array());
                    },
                    cancel: cancelled,
                }),
                { headers: { 'content-type': 'application/json' } },
            ),
        );
        await expect(driver.countIndexedNativeRequest(request(), target)).rejects.toThrow('bounded JSON');
        expect(cancelled).toHaveBeenCalledOnce();
    });

    it('propagates cancellation to the actual SDK endpoint and refuses already cancelled input', async () => {
        const { driver } = countedDriver();
        const endpoint = vi.fn<typeof fetch>(async (_url, options) => {
            const signal = options?.signal;
            if (!signal) throw new Error('Owned SDK count did not provide its abort signal');
            return await new Promise<Response>((_resolve, reject) =>
                signal.addEventListener('abort', () => reject(signal.reason), { once: true }),
            );
        });
        driver.service = driver.service.withOptions({ fetch: endpoint, maxRetries: 0 });
        const controller = new AbortController();
        const counting = countOpenAIIndexedRequest(driver.service, request(), target, controller.signal);
        await vi.waitFor(() => expect(endpoint).toHaveBeenCalledOnce());
        const reason = new Error('Owned native preparation cancelled');
        controller.abort(reason);
        await expect(counting).rejects.toBe(reason);
        await expect(countOpenAIIndexedRequest(driver.service, request(), target, controller.signal)).rejects.toBe(
            reason,
        );
        expect(endpoint).toHaveBeenCalledOnce();
    });

    it('bounds the native envelope and refuses accessor inputs without executing them', () => {
        expect(() => openAIIndexedCountBody({ ...request(), input: 'x'.repeat(32 * 1024 * 1024) }, target)).toThrow(
            'bounded',
        );
        const getter = vi.fn(() => request().input);
        expect(() =>
            openAIIndexedCountBody(
                {
                    model: target.model,
                    get input() {
                        return getter();
                    },
                },
                target,
            ),
        ).toThrow('bounded');
        expect(getter).not.toHaveBeenCalled();
    });
});
