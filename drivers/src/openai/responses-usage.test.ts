import { Providers } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from './index.js';
import { OpenAIDriver } from './openai.js';

class TestResponsesDriver extends OpenAIResponsesDriverBase {
    provider: OpenAIResponsesDriverBase['provider'] = Providers.openai;
    service: OpenAI;

    constructor(create: (request: unknown) => Promise<unknown>) {
        super({});
        this.service = { responses: { create } } as unknown as OpenAI;
    }
}

function response(extra: Partial<OpenAI.Responses.Response> = {}) {
    return {
        id: 'response-1',
        object: 'response',
        created_at: 1,
        model: 'gpt-5',
        status: 'completed',
        output: [
            {
                id: 'msg-1',
                type: 'message',
                role: 'assistant',
                status: 'completed',
                content: [{ type: 'output_text', text: 'answer', annotations: [], logprobs: [] }],
            },
        ],
        output_text: 'answer',
        parallel_tool_calls: true,
        tool_choice: 'auto',
        tools: [],
        error: null,
        incomplete_details: null,
        instructions: null,
        metadata: null,
        temperature: null,
        top_p: null,
        usage: {
            input_tokens: 100,
            output_tokens: 20,
            total_tokens: 120,
            input_tokens_details: {
                cached_tokens: 40,
                cache_write_tokens: 50,
                cache_write_tokens_details: { ttl_12h_tokens: 35, ttl_30m_tokens: 15 },
            },
            output_tokens_details: { reasoning_tokens: 0 },
        },
        ...extra,
    } as unknown as OpenAI.Responses.Response;
}

const toolUsage: OpenAI.Responses.Response.ToolUsage = {
    image_gen: {
        input_tokens: 30,
        input_tokens_details: { image_tokens: 10, text_tokens: 20 },
        output_tokens: 1100,
        output_tokens_details: { image_tokens: 1056, text_tokens: 44 },
        total_tokens: 1130,
        cached_tokens: 5,
    },
    web_search: { num_requests: 2 },
};

async function complete(final: OpenAI.Responses.Response, streaming: boolean) {
    const create = vi.fn(async (request: unknown) =>
        (request as { stream?: boolean }).stream
            ? (async function* () {
                  yield { type: 'response.completed', sequence_number: 1, response: final };
              })()
            : final,
    );
    const driver = new TestResponsesDriver(create);
    const warn = vi.fn();
    driver.logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() };
    const prompt = [{ type: 'message' as const, role: 'user' as const, content: 'question' }];
    const options = { model: 'gpt-5' };
    if (!streaming) return { completion: await driver.requestTextCompletion(prompt, options), warn };
    let token_usage: unknown;
    for await (const chunk of await driver.requestTextCompletionStream(prompt, options)) {
        if (typeof chunk === 'object' && chunk.token_usage) token_usage = chunk.token_usage;
    }
    return { completion: { token_usage }, warn };
}

describe.each([false, true])('OpenAI Responses usage when streaming=%s', (streaming) => {
    it('splits cache writes by retention and reports hosted tool usage apart', async () => {
        const { completion } = await complete(response({ tool_usage: toolUsage }), streaming);
        expect(completion.token_usage).toEqual({
            prompt: 100,
            prompt_cached: 40,
            prompt_cache_write: 50,
            prompt_cache_write_12h: 35,
            prompt_new: 10,
            result: 20,
            total: 120,
            tool_usage: {
                image_generation: { prompt: 30, prompt_cached: 5, result: 1100, result_image: 1056, total: 1130 },
                web_search_requests: 2,
            },
        });
    });

    it('keeps hosted tool usage when the model usage is missing', async () => {
        const { completion } = await complete(response({ usage: undefined, tool_usage: toolUsage }), streaming);
        expect(completion.token_usage).toEqual({
            tool_usage: {
                image_generation: { prompt: 30, prompt_cached: 5, result: 1100, result_image: 1056, total: 1130 },
                web_search_requests: 2,
            },
        });
    });

    it('reports no image tokens for the image tool when the output split is missing', async () => {
        const { output_tokens_details: _omitted, ...imageGen } = toolUsage.image_gen as NonNullable<
            OpenAI.Responses.Response.ToolUsage['image_gen']
        >;
        const { completion } = await complete(
            response({ tool_usage: { image_gen: imageGen, web_search: { num_requests: 0 } } }),
            streaming,
        );
        expect(completion.token_usage).toMatchObject({
            tool_usage: { image_generation: { prompt: 30, prompt_cached: 5, result: 1100, total: 1130 } },
        });
        expect(
            (completion.token_usage as { tool_usage: { image_generation: object } }).tool_usage.image_generation,
        ).not.toHaveProperty('result_image');
    });

    it('leaves out tools that report no use', async () => {
        const unused: OpenAI.Responses.Response.ToolUsage = {
            image_gen: {
                input_tokens: 0,
                input_tokens_details: { image_tokens: 0, text_tokens: 0 },
                output_tokens: 0,
                total_tokens: 0,
            },
            web_search: { num_requests: 0 },
        };
        const { completion } = await complete(response({ tool_usage: unused }), streaming);
        expect(completion.token_usage).not.toHaveProperty('tool_usage');
    });

    it('reports reasoning items the server withheld', async () => {
        const { warn } = await complete(
            response({
                input_transformations: [
                    { id: 'rs_1', action: 'removed', reason: 'model_mismatch', type: 'reasoning' },
                    { id: 'rs_2', action: 'removed', reason: 'model_mismatch', type: 'reasoning' },
                    { id: 'rs_3', action: 'removed', reason: 'turn_boundary', type: 'reasoning' },
                ],
            }),
            streaming,
        );
        expect(warn).toHaveBeenCalledWith(
            {
                provider: Providers.openai,
                model: 'gpt-5',
                transformation_count: 3,
                transformations: { 'reasoning:removed:model_mismatch': 2, 'reasoning:removed:turn_boundary': 1 },
                first_id: 'rs_1',
            },
            '[OpenAI] Replayed reasoning items were withheld from the model',
        );
    });
});

describe('OpenAI Images usage', () => {
    it('reports image tokens and the tokens of the model that authored an SVG', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        vi.spyOn(driver.service.images, 'generate').mockResolvedValue({
            created: 0,
            data: [{ b64_json: 'PHN2Zy8+' }],
            output_format: 'svg',
            usage: {
                input_tokens: 12,
                input_tokens_details: { image_tokens: 0, text_tokens: 12 },
                output_tokens: 300,
                output_tokens_details: { image_tokens: 280, text_tokens: 20 },
                total_tokens: 312,
            },
            svg_usage: {
                model: 'gpt-6-sol',
                input_tokens: 40,
                output_tokens: 900,
                reasoning_tokens: 600,
                total_tokens: 940,
            },
        } as OpenAI.Images.ImagesResponse);

        const completion = await driver.requestImageGeneration([{ role: 'user', content: 'A lighthouse' }], {
            model: 'gpt-image-2.6-flare',
            model_options: { _option_id: 'openai-gpt-image', output_format: 'svg' },
        });

        expect(completion.token_usage).toEqual({
            prompt: 12,
            prompt_new: 12,
            result: 300,
            result_image: 280,
            total: 312,
            tool_usage: { svg: { model: 'gpt-6-sol', prompt: 40, result: 900, total: 940 } },
        });
    });
});
