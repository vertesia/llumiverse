import type { ToolUse } from '@llumiverse/common';
import { describe, expect, test } from 'vitest';
import { accumulateToolUseChunk } from '../src/CompletionStream.js';

type StreamingToolUse = ToolUse<unknown> & { _actual_id?: string };

/**
 * Split a string into N roughly equal pieces, preserving order and content.
 * Used to simulate a provider streaming a single large value across several chunks.
 */
function splitIntoChunks(value: string, pieces: number): string[] {
    const size = Math.ceil(value.length / pieces);
    const out: string[] = [];
    for (let i = 0; i < value.length; i += size) {
        out.push(value.slice(i, i + size));
    }
    return out;
}

/**
 * Drive the streaming tool-use accumulator with an ordered list of chunk fragments
 * (mirrors the per-chunk loop inside DefaultCompletionStream) and return the final tool.
 */
function accumulate(fragments: StreamingToolUse[]): StreamingToolUse | undefined {
    const map = new Map<string, StreamingToolUse>();
    for (const f of fragments) {
        accumulateToolUseChunk(map, f);
    }
    return Array.from(map.values())[0];
}

describe('streaming tool_use thought_signature reassembly', () => {
    // A large, valid base64 signature like Gemini 2.5+/3.x thinking models return on
    // high-effort turns. Length is a multiple of 4 so it is valid, decodable base64.
    const bytes = Uint8Array.from({ length: 9000 }, (_v, i) => (i * 37 + 11) % 256);
    const fullSignature = Buffer.from(bytes).toString('base64');

    test('the fixture signature is valid base64 to begin with', () => {
        expect(fullSignature.length % 4).toBe(0);
        expect(Buffer.from(fullSignature, 'base64')).toEqual(Buffer.from(bytes));
    });

    test('reassembles a signature split across chunks byte-for-byte (valid base64)', () => {
        // Gemini streams the function call across several chunks. The tool name + signature
        // arrive in pieces under the same tool id; arguments accumulate in parallel.
        const pieces = splitIntoChunks(fullSignature, 5);
        const argPieces = splitIntoChunks('{"query":"weather in Paris"}', 5);

        const fragments: StreamingToolUse[] = pieces.map((sigPiece, idx) => ({
            id: 'get_weather',
            tool_name: idx === 0 ? 'get_weather' : '',
            tool_input: argPieces[idx] ?? '',
            thought_signature: sigPiece,
        }));

        const result = accumulate(fragments);

        expect(result).toBeDefined();
        // The reassembled signature must be byte-identical to what the model emitted...
        expect(result?.thought_signature).toBe(fullSignature);
        // ...and therefore still decode as valid base64 (the round-trip Vertex requires).
        expect(() => Buffer.from(result?.thought_signature ?? '', 'base64')).not.toThrow();
        expect(Buffer.from(result?.thought_signature ?? '', 'base64')).toEqual(Buffer.from(bytes));
    });

    test('carries a signature that only appears on a later chunk', () => {
        // First chunk opens the call with no signature; the signature lands on a later chunk.
        const fragments: StreamingToolUse[] = [
            { id: 'plan', tool_name: 'plan', tool_input: '{"steps":' },
            { id: 'plan', tool_name: '', tool_input: '[]}', thought_signature: fullSignature },
        ];

        const result = accumulate(fragments);

        expect(result?.thought_signature).toBe(fullSignature);
    });

    test('preserves a signature delivered whole in a single chunk (no double-encoding)', () => {
        const fragments: StreamingToolUse[] = [
            { id: 'plan', tool_name: 'plan', tool_input: '{}', thought_signature: fullSignature },
        ];

        const result = accumulate(fragments);

        // A single-fragment signature must round-trip unchanged (not concatenated with itself).
        expect(result?.thought_signature).toBe(fullSignature);
    });
});
