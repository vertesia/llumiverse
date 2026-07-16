import { ThinkingLevel } from '@google/genai';
import type { StatelessExecutionOptions } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { geminiThinkingConfig } from './gemini.js';

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
            includeThoughts: false,
            thinkingLevel: ThinkingLevel.MINIMAL,
        });
        expect(geminiThinkingConfig(options('gemini-3.1-pro', { effort: 'medium' }))).toEqual({
            includeThoughts: false,
            thinkingLevel: ThinkingLevel.MEDIUM,
        });
    });

    it('preserves explicitly requested thought inclusion without imposing a thinking level', () => {
        expect(geminiThinkingConfig(options('gemini-3.5-flash', { include_thoughts: true }))).toEqual({
            includeThoughts: true,
        });
    });
});
