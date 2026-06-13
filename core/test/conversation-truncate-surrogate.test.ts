import { describe, expect, test } from 'vitest';
import { truncateLargeTextInConversation } from '../src/conversation-utils.js';

// Matches a lone (unpaired) UTF-16 surrogate: a high surrogate not followed by a
// low surrogate, or a low surrogate not preceded by a high one. A string with a
// lone surrogate is not valid Unicode and cannot be encoded to UTF-8 losslessly,
// so strict JSON parsers (e.g. Vertex/Gemini) reject the request body as
// "The input data is not valid json".
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

// truncateLargeTextInConversation uses maxChars = textMaxTokens * CHARS_PER_TOKEN
// (CHARS_PER_TOKEN = 4).
const CHARS_PER_TOKEN = 4;

describe('truncateLargeTextInConversation — surrogate safety', () => {
    test('does not leave a lone surrogate when the boundary splits a surrogate pair', () => {
        const maxTokens = 10;
        const maxChars = maxTokens * CHARS_PER_TOKEN; // 40

        // Put a non-BMP emoji (🎉 = U+1F389 = '🎉', a surrogate pair)
        // so its HIGH surrogate sits at index maxChars-1 and its LOW surrogate at
        // index maxChars. substring(0, maxChars) keeps the high, drops the low.
        const input = `${'a'.repeat(maxChars - 1)}🎉${'b'.repeat(50)}`;
        expect(input.isWellFormed()).toBe(true); // sanity: the source is valid
        expect(input.length).toBeGreaterThan(maxChars);

        const out = truncateLargeTextInConversation(input, { textMaxTokens: maxTokens }) as string;

        // The truncated text must remain valid UTF-16 — no lone surrogate.
        expect(LONE_SURROGATE.test(out)).toBe(false);
        expect(out.isWellFormed()).toBe(true);

        // And it must survive a UTF-8 round-trip unchanged (an HTTP/JSON body is
        // UTF-8). A lone surrogate becomes U+FFFD on round-trip, so this would
        // differ — exactly the corruption Vertex rejects.
        expect(Buffer.from(out, 'utf8').toString('utf8')).toBe(out);
    });

    test('control: truncating between BMP characters stays well-formed', () => {
        const out = truncateLargeTextInConversation('a'.repeat(200), { textMaxTokens: 10 }) as string;
        expect(LONE_SURROGATE.test(out)).toBe(false);
        expect(out.isWellFormed()).toBe(true);
    });
});
