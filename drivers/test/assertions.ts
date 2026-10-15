import {
    type AbstractDriver,
    type CompletionStream,
    type ExecutionResponse,
    extractAndParseJSON,
} from '@llumiverse/core';
import { expect } from 'vitest';
import { completionResultToString, parseCompletionResultsToJson } from './utils.js';

/** Answer text without reasoning, so a response that only "thought" does not count as an answer. */
export function answerText(r: Pick<ExecutionResponse, 'result'>): string {
    return r.result
        .filter((result) => result.type !== 'thoughts')
        .map(completionResultToString)
        .join('');
}

/** Points at max_tokens when a failed check is explained by a truncated response. */
function truncationHint(r: ExecutionResponse): string {
    return r.finish_reason === 'length'
        ? ' The response was cut off at max_tokens (finish_reason "length"): raise max_tokens for this model.'
        : '';
}

export function assertCompletionOk(r: ExecutionResponse, model?: string, driver?: AbstractDriver) {
    expect(r.error, truncationHint(r)).toBeFalsy();
    expect(r.prompt).toBeTruthy();
    //TODO: This just checks for existence of the object,
    //could do with more thorough test however not all models support token_usage.
    //Only create the object when there is meaningful information you want to interpret as a pass.
    if (!(driver?.provider === 'bedrock' && model?.includes('mistral'))) {
        //Skip if bedrock:mistral, token_usage not supported.
        expect(r.token_usage).toBeTruthy();
    }
    expect(r.finish_reason).toBeTruthy();
    expect(answerText(r).length, `Expected a non-empty answer.${truncationHint(r)}`).toBeGreaterThan(2);
}

export async function assertStreamingCompletionOk(stream: CompletionStream, jsonMode: boolean = false) {
    const out: string[] = [];
    for await (const chunk of stream) {
        out.push(chunk);
        console.log(chunk);
    }
    console.log(out.join(''));
    const r = stream.completion as ExecutionResponse;
    expect(r.error, truncationHint(r)).toBeFalsy();
    const jsonObject = jsonMode ? extractAndParseJSON(out.join('')) : undefined;
    const jsonResult = jsonMode ? parseCompletionResultsToJson(r.result) : undefined;
    console.log(jsonObject);
    console.log(jsonResult);
    if (jsonMode) {
        expect(jsonResult).toStrictEqual(jsonObject);
    }

    expect(r.prompt).toBeTruthy();
    expect(r.token_usage).toBeTruthy();
    expect(r.finish_reason).toBeTruthy();
    expect(answerText(r).length, `Expected a non-empty answer.${truncationHint(r)}`).toBeGreaterThan(2);

    return out;
}
