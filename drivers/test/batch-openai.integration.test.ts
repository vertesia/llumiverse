/**
 * Live integration test: OpenAI batch inference vs synchronous execution on a real
 * document. Renders the first N pages to JPEG, transcribes each to markdown via BOTH
 * the synchronous path and the async Batch API, and reports the comparison.
 *
 * Gated — runs only when set:
 *   OPENAI_API_KEY
 *   BATCH_TEST_PDF       path to the PDF (e.g. the Fed report)
 *   OPENAI_BATCH_MODEL   default gpt-5-mini
 *   BATCH_TEST_PAGES     default 4
 *   BATCH_OUT_DIR        default $TMPDIR/batch-compare-openai
 *
 * Run: set -a; source ai-intake/.env; set +a; \
 *   BATCH_TEST_PDF=… pnpm --filter @llumiverse/drivers exec vitest run test/batch-openai.integration.test.ts
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
    Base64DataSource,
    type BatchInferenceRequestItem,
    type CompletionResult,
    type ExecutionOptions,
    type PromptSegment,
} from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { OpenAIDriver } from '../src/index.js';

const API_KEY = process.env.OPENAI_API_KEY;
const PDF = process.env.BATCH_TEST_PDF;
const MODEL = process.env.OPENAI_BATCH_MODEL ?? 'gpt-5-mini';
const PAGES = Number(process.env.BATCH_TEST_PAGES ?? '4');
const POLL_MS = 30_000;
const TIMEOUT_MS = 90 * 60 * 1000;

const gated = Boolean(API_KEY && PDF && existsSync(PDF));

function resultText(result: CompletionResult[]): string {
    return result.map((r) => (r.type === 'text' ? r.value : r.type === 'json' ? JSON.stringify(r.value) : '')).join('');
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe.skipIf(!gated)('OpenAI batch vs sync — document transcription', () => {
    it(
        'transcribes the same pages via batch and sync and reports the comparison',
        async () => {
            const dir = mkdtempSync(join(tmpdir(), 'batchpdf-oai-'));
            execFileSync('pdftoppm', [
                '-f',
                '1',
                '-l',
                String(PAGES),
                '-r',
                '150',
                '-jpeg',
                PDF as string,
                join(dir, 'p'),
            ]);
            const files = readdirSync(dir)
                .filter((f) => f.endsWith('.jpg'))
                .sort();
            expect(files.length).toBeGreaterThan(0);

            const driver = new OpenAIDriver({ apiKey: API_KEY as string });
            const PROMPT =
                'Convert this document page image to clean GitHub-flavored Markdown. Output only the markdown, with no preamble.';

            const requests: BatchInferenceRequestItem[] = files.map((f, i) => ({
                custom_id: `page-${i + 1}`,
                segments: [
                    { role: 'system', content: PROMPT },
                    {
                        role: 'user',
                        content: '',
                        files: [new Base64DataSource(f, 'image/jpeg', readFileSync(join(dir, f)).toString('base64'))],
                    },
                ] as PromptSegment[],
                options: { model: MODEL, model_options: { max_tokens: 4096 } } as ExecutionOptions,
            }));

            // --- synchronous path (tolerate gpt-5-mini's occasional empty completions) ---
            const syncStart = Date.now();
            const sync = new Map<string, string>();
            let syncErrors = 0;
            for (const r of requests) {
                try {
                    const res = await driver.execute(r.segments, r.options);
                    sync.set(r.custom_id, resultText(res.result));
                } catch (e) {
                    syncErrors++;
                    sync.set(r.custom_id, '');
                    // eslint-disable-next-line no-console
                    console.log(`[openai-sync] ${r.custom_id} failed: ${(e as Error).message}`);
                }
            }
            const syncMs = Date.now() - syncStart;

            // --- batch path ---
            const submitStart = Date.now();
            const job = await driver.startBatchInference(requests, { name: 'doc-batch-test' });
            // eslint-disable-next-line no-console
            console.log(`[openai-batch] submitted job=${job.id} status=${job.status}`);
            let status = job;
            while (status.status === 'queued' || status.status === 'running') {
                await sleep(POLL_MS);
                status = await driver.getBatchInferenceJob(job.id);
                // eslint-disable-next-line no-console
                console.log(`[openai-batch] status=${status.status}`);
            }
            const batchMs = Date.now() - submitStart;
            expect(status.status).toBe('succeeded');

            const results = await driver.getBatchInferenceResults(job.id);
            const batch = new Map(results.map((r) => [r.custom_id, resultText(r.result ?? [])]));
            const batchTokens = new Map(results.map((r) => [r.custom_id, r.token_usage]));

            expect(batch.size).toBe(requests.length);
            const outDir = process.env.BATCH_OUT_DIR ?? join(tmpdir(), 'batch-compare-openai');
            mkdirSync(outDir, { recursive: true });
            const perPage: Array<Record<string, unknown>> = [];
            let batchNonEmpty = 0;
            for (const r of requests) {
                const s = (sync.get(r.custom_id) ?? '').trim();
                const b = (batch.get(r.custom_id) ?? '').trim();
                if (b.length > 0) {
                    batchNonEmpty++;
                }
                writeFileSync(join(outDir, `${r.custom_id}.sync.md`), s);
                writeFileSync(join(outDir, `${r.custom_id}.batch.md`), b);
                perPage.push({
                    custom_id: r.custom_id,
                    sync_chars: s.length,
                    batch_chars: b.length,
                    first120_exact: s.slice(0, 120) === b.slice(0, 120),
                    identical: s === b,
                    batch_tokens: batchTokens.get(r.custom_id),
                });
            }
            const summary = {
                model: MODEL,
                pages: requests.length,
                sync_ms: syncMs,
                sync_errors: syncErrors,
                batch_ms: batchMs,
                batch_non_empty: batchNonEmpty,
                batch_job_id: job.id,
                per_page: perPage,
            };
            writeFileSync(join(outDir, 'summary.json'), JSON.stringify(summary, null, 2));
            // eslint-disable-next-line no-console
            console.log(`[openai-batch-vs-sync] ${JSON.stringify(summary)}`);
            expect(results.length).toBe(requests.length);
        },
        TIMEOUT_MS,
    );
});
