/**
 * An incomplete Responses API result (max_output_tokens or content_filter) can stop inside its last function call.
 * The completion must report that stop rather than `tool_use`, and must not return a call whose arguments were cut
 * off as if it had empty input.
 */

import type OpenAI from 'openai';
import { describe, expect, it } from 'vitest';
import { OpenAIDriver } from './openai.js';

function incompleteResponse(reason: string): OpenAI.Responses.Response {
    return {
        id: 'resp-1',
        status: 'incomplete',
        incomplete_details: { reason },
        output: [
            { type: 'function_call', call_id: 'call_1', name: 'search', arguments: '{"query":"expenses 2025"}' },
            { type: 'function_call', call_id: 'call_2', name: 'delete_rows', arguments: '{"where":"id=' },
        ],
    } as unknown as OpenAI.Responses.Response;
}

describe('OpenAI Responses cut off inside a function call', () => {
    const driver = new OpenAIDriver({ apiKey: 'test-key' });

    it.each([
        ['content_filter', 'content_filter'],
        ['max_output_tokens', 'length'],
    ])('reports %s and leaves out the cut-off call', (reason, finishReason) => {
        const completion = driver.extractDataFromResponse({ model: 'gpt-test' }, incompleteResponse(reason));

        expect(completion.finish_reason).toBe(finishReason);
        expect(completion.tool_use).toEqual([
            { id: 'call_1', tool_name: 'search', tool_input: { query: 'expenses 2025' } },
        ]);
    });

    it('still reports tool_use for a completed response with function calls', () => {
        const completed = {
            ...incompleteResponse('content_filter'),
            status: 'completed',
            incomplete_details: null,
        } as unknown as OpenAI.Responses.Response;

        expect(driver.extractDataFromResponse({ model: 'gpt-test' }, completed).finish_reason).toBe('tool_use');
    });
});
