
import { Bedrock } from '@aws-sdk/client-bedrock';
import { AbstractDriver, CompletionStream, ExecutionResponse, extractAndParseJSON, parseResultAsJson, resultAsString } from '@llumiverse/core';
import { expect } from "vitest";
import { BedrockDriver } from '../src';

export function assertCompletionOk(r: ExecutionResponse, model?: string, driver?: AbstractDriver) {
    expect(r.error).toBeFalsy();
    expect(r.prompt).toBeTruthy();
    //TODO: This just checks for existence of the object,
    //could do with more thorough test however not all models support token_usage.
    //Only create the object when there is meaningful information you want to interpret as a pass.
    if (!(driver?.provider == 'bedrock' && model?.includes("mistral"))) { //Skip if bedrock:mistral, token_usage not supported.
        expect(r.token_usage).toBeTruthy();
    }
    expect(r.finish_reason).toBeTruthy();
    //if r.result is string, it should be longer than 2
    const stringResult = r.result.map(resultAsString).join();
    expect(stringResult.length).toBeGreaterThan(2);
}

export async function assertStreamingCompletionOk(stream: CompletionStream, jsonMode: boolean=false) {

    const out: string[] = []
    for await (const chunk of stream) {
        out.push(chunk)
        console.log(chunk)
    }
    console.log(out.join());
    const r = stream.completion as ExecutionResponse;
    const jsonObject = jsonMode ? extractAndParseJSON(out.join()) : undefined;
    const jsonResult = jsonMode ? parseResultAsJson(r.result) : undefined;
    console.log(jsonObject);
    console.log(jsonResult);
    if (jsonMode) {
        expect(jsonResult).toStrictEqual(jsonObject);
    }
    
    expect(r.error).toBeFalsy();
    expect(r.prompt).toBeTruthy();
    expect(r.token_usage).toBeTruthy();
    expect(r.finish_reason).toBeTruthy();
    const stringResult = r.result.map(resultAsString).join();
    expect(stringResult.length).toBeGreaterThan(2);

    return out;
}
