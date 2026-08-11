import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { parseOpenAIBatchErrorLine } from './batch.js';
import { OpenAIDriver } from './openai.js';

function outputLine(custom_id: string, text: string): string {
    return JSON.stringify({ custom_id, response: { status_code: 200, body: { output_text: text } } });
}

function errorLine(custom_id: string, message: string): string {
    return JSON.stringify({
        custom_id,
        response: { status_code: 400, body: { error: { message, type: 'invalid_request_error' } } },
        error: null,
    });
}

function mockDriver(job: {
    output_file_id?: string | null;
    error_file_id?: string | null;
    files?: Record<string, string>;
}): OpenAIDriver {
    const driver = new OpenAIDriver({ apiKey: 'test-key' });
    const retrieve = vi.fn().mockResolvedValue({
        id: 'batch-1',
        status: 'completed',
        output_file_id: job.output_file_id ?? null,
        error_file_id: job.error_file_id ?? null,
    });
    const content = vi.fn().mockImplementation(async (fileId: string) => ({
        text: async () => job.files?.[fileId] ?? '',
    }));
    driver.service = {
        batches: { retrieve },
        files: { content },
    } as unknown as OpenAI;
    return driver;
}

describe('parseOpenAIBatchErrorLine', () => {
    it('extracts the response body error message', () => {
        const item = parseOpenAIBatchErrorLine(errorLine('doc-1', 'model overloaded'), 0);
        expect(item).toEqual({ custom_id: 'doc-1', error: 'model overloaded' });
    });

    it('falls back to the top-level error object', () => {
        const item = parseOpenAIBatchErrorLine(
            JSON.stringify({ custom_id: 'doc-2', error: { code: 'server_error', message: 'boom' } }),
            0,
        );
        expect(item?.custom_id).toBe('doc-2');
        expect(item?.error).toContain('boom');
    });

    it('reports the status code when no error payload is present', () => {
        const item = parseOpenAIBatchErrorLine(
            JSON.stringify({ custom_id: 'doc-3', response: { status_code: 500 } }),
            0,
        );
        expect(item).toEqual({ custom_id: 'doc-3', error: 'request failed with status 500' });
    });

    it('ignores blank and invalid lines', () => {
        expect(parseOpenAIBatchErrorLine('   ', 0)).toBeUndefined();
        expect(parseOpenAIBatchErrorLine('not json', 0)).toBeUndefined();
    });
});

describe('OpenAI getBatchInferenceResults', () => {
    it('merges error-file items with output-file items', async () => {
        const driver = mockDriver({
            output_file_id: 'file-out',
            error_file_id: 'file-err',
            files: {
                'file-out': [outputLine('doc-1', 'first'), outputLine('doc-2', 'second')].join('\n'),
                'file-err': errorLine('doc-3', 'schema validation failed'),
            },
        });
        const items = await driver.getBatchInferenceResults('batch-1');
        expect(items).toHaveLength(3);
        expect(items.find((i) => i.custom_id === 'doc-1')?.result).toEqual([{ type: 'text', value: 'first' }]);
        expect(items.find((i) => i.custom_id === 'doc-3')).toEqual({
            custom_id: 'doc-3',
            error: 'schema validation failed',
        });
    });

    it('returns error items when all requests failed (no output file)', async () => {
        const driver = mockDriver({
            output_file_id: null,
            error_file_id: 'file-err',
            files: { 'file-err': [errorLine('doc-1', 'bad request'), errorLine('doc-2', 'too long')].join('\n') },
        });
        const items = await driver.getBatchInferenceResults('batch-1');
        expect(items).toEqual([
            { custom_id: 'doc-1', error: 'bad request' },
            { custom_id: 'doc-2', error: 'too long' },
        ]);
    });

    it('prefers the output-file item when a custom_id appears in both files', async () => {
        const driver = mockDriver({
            output_file_id: 'file-out',
            error_file_id: 'file-err',
            files: {
                'file-out': outputLine('doc-1', 'kept'),
                'file-err': errorLine('doc-1', 'stale duplicate'),
            },
        });
        const items = await driver.getBatchInferenceResults('batch-1');
        expect(items).toEqual([
            { custom_id: 'doc-1', result: [{ type: 'text', value: 'kept' }], token_usage: undefined },
        ]);
    });

    it('throws only when the job has neither output nor error file', async () => {
        const driver = mockDriver({ output_file_id: null, error_file_id: null });
        await expect(driver.getBatchInferenceResults('batch-1')).rejects.toThrow(
            '[openai] batch job has no output or error file',
        );
    });
});
