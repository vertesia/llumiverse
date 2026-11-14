import { describe, expect, it } from 'vitest';
import { ClaudeModelDefinition } from './models/claude.js';
import { GeminiModelDefinition } from './models/gemini.js';
import { getModelDefinition } from './models.js';

describe('Vertex model routing', () => {
    it('routes a bare Claude id to the Claude definition instead of the Gemini fallback', () => {
        for (const id of ['claude-opus-5-5', 'claude-sonnet-4-6', 'claude-opus-9-1@20990101']) {
            expect(getModelDefinition(id)).toBeInstanceOf(ClaudeModelDefinition);
        }
    });

    it('keeps publisher paths and bare Gemini ids on their existing definitions', () => {
        expect(getModelDefinition('locations/global/publishers/anthropic/models/claude-opus-5-5')).toBeInstanceOf(
            ClaudeModelDefinition,
        );
        expect(getModelDefinition('publishers/google/models/gemini-2.5-flash')).toBeInstanceOf(GeminiModelDefinition);
        expect(getModelDefinition('gemini-2.5-flash')).toBeInstanceOf(GeminiModelDefinition);
    });
});
