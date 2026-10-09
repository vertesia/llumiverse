import { ThinkingLevel } from '@google/genai';
import type { StatelessExecutionOptions } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { geminiThinkingConfig, getGeminiPayload } from './gemini.js';

function options(model: string, model_options?: Record<string, unknown>): StatelessExecutionOptions {
    return { model, model_options } as StatelessExecutionOptions;
}

describe('Gemini thinking configuration', () => {
    it('leaves thinking undefined when the caller did not configure it', () => {
        expect(geminiThinkingConfig(options('gemini-3.5-flash'))).toBeUndefined();
        expect(geminiThinkingConfig(options('gemini-2.5-pro'))).toBeUndefined();
    });

    it('maps current Gemini 3 effort levels', () => {
        expect(geminiThinkingConfig(options('gemini-3.5-flash', { effort: 'minimal' }))).toEqual({
            includeThoughts: true,
            thinkingLevel: ThinkingLevel.MINIMAL,
        });
        expect(geminiThinkingConfig(options('gemini-3.1-pro', { effort: 'medium' }))).toEqual({
            includeThoughts: true,
            thinkingLevel: ThinkingLevel.MEDIUM,
        });
    });

    it('passes through caller effort even when advisory metadata does not offer it', () => {
        expect(geminiThinkingConfig(options('gemini-3.1-flash-image', { effort: 'low' }))).toEqual({
            includeThoughts: true,
            thinkingLevel: ThinkingLevel.LOW,
        });
        expect(geminiThinkingConfig(options('gemini-3-pro-image', { effort: 'minimal' }))).toEqual({
            includeThoughts: true,
            thinkingLevel: ThinkingLevel.MINIMAL,
        });
    });

    it('preserves explicitly requested thought inclusion without imposing a thinking level', () => {
        expect(geminiThinkingConfig(options('gemini-3.5-flash', { include_thoughts: true }))).toEqual({
            includeThoughts: true,
        });
    });
});

describe('explicit Gemini thinking controls', () => {
    it.each([0, -1, 8192])('preserves budget %s and disabled thought inclusion', (thinking_budget_tokens) => {
        expect(
            geminiThinkingConfig(
                options('gemini-2.5-flash', {
                    thinking_budget_tokens,
                    include_thoughts: false,
                    effort: 'high',
                }),
            ),
        ).toEqual({ includeThoughts: false, thinkingBudget: thinking_budget_tokens });
    });

    it('preserves explicit thought inclusion with a zero budget for provider validation', () => {
        expect(
            geminiThinkingConfig(
                options('gemini-2.5-flash', {
                    thinking_budget_tokens: 0,
                    include_thoughts: true,
                }),
            ),
        ).toEqual({ includeThoughts: true, thinkingBudget: 0 });
    });

    it('honors thought exclusion with an explicit thinking level', () => {
        expect(
            geminiThinkingConfig(
                options('gemini-3.8-flash', {
                    thinking_level: ThinkingLevel.LOW,
                    include_thoughts: false,
                }),
            ),
        ).toEqual({ includeThoughts: false, thinkingLevel: ThinkingLevel.LOW });
    });
});

describe('Gemini Flash generation parameters', () => {
    it.each(['gemini-3.7-flash', 'publishers/google/models/gemini-3.8-flash-cyber', 'gemini-4.0-flash'])(
        'preserves caller parameters for provider validation on %s',
        (model) => {
            const payload = getGeminiPayload(
                options(model, {
                    temperature: 0.7,
                    top_p: 0.9,
                    top_k: 10,
                    presence_penalty: 0.5,
                    frequency_penalty: 0.5,
                    max_tokens: 512,
                    effort: 'low',
                }),
                { contents: [{ role: 'user', parts: [{ text: 'Hello' }] }] },
            );
            const config = JSON.parse(JSON.stringify(payload.config));
            expect(config).not.toHaveProperty('candidateCount');
            expect(config).toMatchObject({
                temperature: 0.7,
                topP: 0.9,
                topK: 10,
                presencePenalty: 0.5,
                frequencyPenalty: 0.5,
            });
            expect(config).toMatchObject({ maxOutputTokens: 512, thinkingConfig: { thinkingLevel: 'LOW' } });
        },
    );

    it('preserves sampling for earlier Flash models', () => {
        const payload = getGeminiPayload(options('gemini-3.6-flash', { temperature: 0.7, top_p: 0.9 }), {
            contents: [],
        });
        expect(payload.config).toMatchObject({ candidateCount: 1, temperature: 0.7, topP: 0.9 });
    });
});
