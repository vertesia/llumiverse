import { type DataSource, DECISION_INPUT_NOT_SUPPORTED, LlumiverseError, PromptRole } from '@llumiverse/common';
import type OpenAI from 'openai';
import { describe, expect, it } from 'vitest';
import {
    buildOpenAIDecisionRequest,
    fromOpenAIDecision,
    toOpenAIDecisionInput,
    toOpenAIDecisionQuestions,
} from './decisions.js';

function source(mime_type: string, text: string, name = 'file'): DataSource {
    return {
        name,
        mime_type,
        getURL: async () => `memory://${name}`,
        getURI: async () => `memory://${name}`,
        getStream: async () =>
            new ReadableStream<Uint8Array>({
                start(controller) {
                    controller.enqueue(new TextEncoder().encode(text));
                    controller.close();
                },
            }),
    };
}

describe('OpenAI decision input', () => {
    it('puts system and safety text first, then the conversation', async () => {
        const input = await toOpenAIDecisionInput(
            [
                { role: PromptRole.user, content: 'Classify this.' },
                { role: PromptRole.system, content: 'You classify documents.' },
                { role: PromptRole.safety, content: 'Never guess.' },
                { role: PromptRole.assistant, content: 'Earlier answer' },
                { role: PromptRole.negative, content: 'ignored' },
            ],
            'openai',
            'gpt-6-luna',
        );
        expect(input).toEqual([
            { role: 'system', content: 'You classify documents.' },
            { role: 'system', content: 'IMPORTANT: Never guess.' },
            { role: 'user', content: [{ type: 'input_text', text: 'Classify this.' }] },
            { role: 'assistant', content: 'Earlier answer' },
        ]);
    });

    it('sends images as data URLs and inlines text files', async () => {
        const input = await toOpenAIDecisionInput(
            [
                {
                    role: PromptRole.user,
                    content: 'Which type?',
                    files: [source('image/png', 'png'), source('text/plain', 'Invoice #12')],
                },
            ],
            'openai',
            'gpt-6-luna',
        );
        expect(input).toEqual([
            {
                role: 'user',
                content: [
                    { type: 'input_image', image_url: `data:image/png;base64,${btoa('png')}` },
                    { type: 'input_text', text: 'Invoice #12' },
                    { type: 'input_text', text: 'Which type?' },
                ],
            },
        ]);
    });

    it('maps tool results to function call outputs', async () => {
        const input = await toOpenAIDecisionInput(
            [{ role: PromptRole.tool, content: 'deleted 3 rows', tool_use_id: 'call_1' }],
            'openai',
            'gpt-6-luna',
        );
        expect(input).toEqual([{ type: 'function_call_output', call_id: 'call_1', output: 'deleted 3 rows' }]);
    });

    it('rejects a PDF with a typed, non-retryable error instead of dropping it', async () => {
        const error = await toOpenAIDecisionInput(
            [{ role: PromptRole.user, content: 'Which type?', files: [source('application/pdf', '%PDF', 'a.pdf')] }],
            'openai',
            'gpt-6-luna',
        ).catch((err: unknown) => err);
        expect(error).toBeInstanceOf(LlumiverseError);
        expect((error as LlumiverseError).name).toBe(DECISION_INPUT_NOT_SUPPORTED);
        expect((error as LlumiverseError).retryable).toBe(false);
        expect((error as LlumiverseError).message).toContain('application/pdf');
    });
});

describe('OpenAI decision questions', () => {
    it('translates boolean to predicate and keeps omitted descriptions omitted', () => {
        expect(
            toOpenAIDecisionQuestions([
                { type: 'boolean', name: 'english', instructions: 'Is the text in English?' },
                {
                    type: 'choice',
                    name: 'document_type',
                    instructions: 'Which type?',
                    options: [{ value: 'invoice', description: 'A bill' }, { value: 'other' }],
                },
                {
                    type: 'score',
                    name: 'risk',
                    instructions: 'How risky?',
                    levels: [{ label: 'low' }, { label: 'high' }],
                },
            ]),
        ).toEqual([
            { type: 'predicate', name: 'english', instructions: 'Is the text in English?' },
            {
                type: 'choice',
                name: 'document_type',
                instructions: 'Which type?',
                choices: [{ value: 'invoice', description: 'A bill' }, { value: 'other' }],
            },
            { type: 'score', name: 'risk', instructions: 'How risky?', levels: [{ label: 'low' }, { label: 'high' }] },
        ]);
    });

    it('builds the request with the caller model unchanged', async () => {
        const request = await buildOpenAIDecisionRequest(
            {
                model: 'gpt-6-luna-2026-10-22',
                input: [{ role: PromptRole.user, content: 'Hello' }],
                questions: [{ type: 'boolean', name: 'greeting', instructions: 'Is it a greeting?' }],
            },
            'openai',
        );
        expect(request.model).toBe('gpt-6-luna-2026-10-22');
        expect(request.questions).toHaveLength(1);
    });
});

describe('OpenAI decision answers', () => {
    const decision: OpenAI.Decisions.Decision = {
        model: 'gpt-6-luna',
        answers: [
            { type: 'predicate', name: 'english', probability: 0.97 },
            {
                type: 'choice',
                name: 'document_type',
                choice: 'invoice',
                confidence: 0.93,
                probabilities: [
                    { value: 'invoice', probability: 0.95 },
                    { value: 'other', probability: 0.05 },
                ],
            },
            {
                type: 'score',
                name: 'risk',
                score: 1,
                confidence: 0.8,
                probabilities: [
                    { label: 'low', value: 0, probability: 0.15 },
                    { label: 'high', value: 1, probability: 0.85 },
                ],
            },
            { type: 'refusal', name: null },
        ],
        usage: {
            input_tokens: 1200,
            input_tokens_details: { cached_tokens: 1000, cache_write_tokens: 0 },
            output_tokens: 0,
            output_tokens_details: { reasoning_tokens: 0 },
            total_tokens: 1200,
        },
    };

    it('keys probabilities by option value or level label and names refusals by position', () => {
        const result = fromOpenAIDecision(decision, [
            { type: 'boolean', name: 'english', instructions: 'English?' },
            { type: 'choice', name: 'document_type', instructions: 'Type?', options: [] },
            { type: 'score', name: 'risk', instructions: 'Risk?', levels: [] },
            { type: 'boolean', name: 'unsafe', instructions: 'Unsafe?' },
        ]);
        expect(result.answers).toEqual([
            { type: 'boolean', name: 'english', probability: 0.97 },
            {
                type: 'choice',
                name: 'document_type',
                value: 'invoice',
                confidence: 0.93,
                probabilities: { invoice: 0.95, other: 0.05 },
            },
            {
                type: 'score',
                name: 'risk',
                score: 1,
                label: 'high',
                confidence: 0.8,
                probabilities: { low: 0.15, high: 0.85 },
            },
            { type: 'refusal', name: 'unsafe' },
        ]);
    });

    it('reports cached input apart from new input', () => {
        expect(fromOpenAIDecision(decision, []).token_usage).toEqual({
            prompt: 1200,
            prompt_cached: 1000,
            prompt_cache_write: undefined,
            prompt_new: 200,
            result: 0,
            total: 1200,
        });
    });
});
