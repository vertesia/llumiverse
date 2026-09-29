/**
 * The streaming accumulator merges tool-use fragments by `id`. Providers without call ids
 * (Gemini keys a call by its function name) must therefore stream each call under a unique
 * key and carry the provider-facing id in `_actual_id`, otherwise parallel calls to one tool
 * collapse into a single call and the follow-up request answers fewer calls than the model made.
 */

import { describe, expect, it } from 'vitest';
import { accumulateToolUseChunk, finalizeStreamingToolUse } from './CompletionStream.js';

type StreamingToolUse = Parameters<typeof accumulateToolUseChunk>[1];

const parallelCalls = (): StreamingToolUse[] => [
    { id: 'create_document#1', _actual_id: 'create_document', tool_name: 'create_document', tool_input: { name: 'A' } },
    { id: 'create_document#2', _actual_id: 'create_document', tool_name: 'create_document', tool_input: { name: 'B' } },
    { id: 'create_document#3', _actual_id: 'create_document', tool_name: 'create_document', tool_input: { name: 'C' } },
];

describe('streaming tool-use accumulation for calls that share a provider id', () => {
    it('collapses calls streamed under one shared id (the shape that produced the Vertex 400)', () => {
        const accumulated = new Map<string, StreamingToolUse>();
        for (const call of parallelCalls()) {
            accumulateToolUseChunk(accumulated, { ...call, id: 'create_document', _actual_id: undefined });
        }
        // Documents the failure mode: three calls merged into one, with the inputs folded together.
        expect(accumulated.size).toBe(1);
        expect(accumulated.get('create_document')?.tool_input).toEqual({ name: 'C' });
    });

    it('keeps every call when each is streamed under its own key and restores the shared id', () => {
        const accumulated = new Map<string, StreamingToolUse>();
        for (const call of parallelCalls()) {
            accumulateToolUseChunk(accumulated, call);
        }
        expect(accumulated.size).toBe(3);

        const finalized = finalizeStreamingToolUse(Array.from(accumulated.values()), 'tool_use', {
            provider: 'vertexai',
            model: 'gemini-2.5-flash',
        });
        expect(finalized).toEqual([
            { id: 'create_document', tool_name: 'create_document', tool_input: { name: 'A' } },
            { id: 'create_document', tool_name: 'create_document', tool_input: { name: 'B' } },
            { id: 'create_document', tool_name: 'create_document', tool_input: { name: 'C' } },
        ]);
    });
});
