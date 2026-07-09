/**
 * Live integration test: Vertex batch inference vs synchronous execution on a real
 * document (the Fed report). Renders the first N pages to JPEG, transcribes each to
 * markdown via BOTH the synchronous path and the async batch path, and reports
 * per-page output, tokens and wall-clock.
 *
 * Gated — runs only when these are set (and GCP ADC is available):
 *   GOOGLE_PROJECT_ID   e.g. vertesia-dev
 *   GOOGLE_REGION       default us-central1
 *   BATCH_BUCKET        a GCS bucket the caller can write to (name or gs://bucket/prefix)
 *   BATCH_TEST_PDF      path to the PDF (e.g. the Fed report)
 *   BATCH_TEST_MODEL    default gemini-2.5-flash-lite
 *   BATCH_TEST_PAGES    default 4
 *   BATCH_OUT_DIR       where to write the sync/batch outputs + summary.json
 *
 * Run: BATCH_TEST_PDF=… BATCH_BUCKET=… pnpm --filter @llumiverse/drivers exec vitest run test/batch-vertex.integration.test.ts
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
import { VertexAIDriver } from '../src/index.js';

const PROJECT = process.env.GOOGLE_PROJECT_ID;
const REGION = process.env.GOOGLE_REGION ?? 'us-central1';
const BUCKET = process.env.BATCH_BUCKET;
const PDF = process.env.BATCH_TEST_PDF;
const MODEL = process.env.BATCH_TEST_MODEL ?? 'gemini-2.5-flash-lite';
const PAGES = Number(process.env.BATCH_TEST_PAGES ?? '4');
const POLL_MS = 20_000;
const TIMEOUT_MS = 60 * 60 * 1000;

const gated = Boolean(PROJECT && BUCKET && PDF && existsSync(PDF));

function resultText(result: CompletionResult[]): string {
    return result.map((r) => (r.type === 'text' ? r.value : r.type === 'json' ? JSON.stringify(r.value) : '')).join('');
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe.skipIf(!gated)('Vertex batch vs sync — document transcription', () => {
    it(
        'transcribes the same pages via batch and sync and reports the comparison',
        async () => {
            const dir = mkdtempSync(join(tmpdir(), 'batchpdf-'));
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

            const driver = new VertexAIDriver({ project: PROJECT as string, region: REGION, batch_bucket: BUCKET });
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
                options: { model: MODEL, model_options: { max_tokens: 4096, temperature: 0 } } as ExecutionOptions,
            }));

            // --- synchronous path ---
            const syncStart = Date.now();
            const sync = new Map<string, string>();
            for (const r of requests) {
                const res = await driver.execute(r.segments, r.options);
                sync.set(r.custom_id, resultText(res.result));
            }
            const syncMs = Date.now() - syncStart;

            // --- batch path ---
            const submitStart = Date.now();
            const job = await driver.startBatchInference(requests, { name: 'doc-batch-test' });
            // eslint-disable-next-line no-console
            console.log(`[batch] submitted job=${job.id} status=${job.status}`);
            let status = job;
            while (status.status === 'queued' || status.status === 'running') {
                await sleep(POLL_MS);
                status = await driver.getBatchInferenceJob(job.id);
                // eslint-disable-next-line no-console
                console.log(`[batch] status=${status.status}`);
            }
            const batchMs = Date.now() - submitStart;
            expect(status.status).toBe('succeeded');

            const results = await driver.getBatchInferenceResults(job.id);
            const batch = new Map(results.map((r) => [r.custom_id, resultText(r.result ?? [])]));
            const batchTokens = new Map(results.map((r) => [r.custom_id, r.token_usage]));

            // --- compare ---
            expect(batch.size).toBe(requests.length);
            const outDir = process.env.BATCH_OUT_DIR ?? join(tmpdir(), 'batch-compare');
            mkdirSync(outDir, { recursive: true });
            let firstLineMatches = 0;
            const perPage: Array<Record<string, unknown>> = [];
            for (const r of requests) {
                const s = (sync.get(r.custom_id) ?? '').trim();
                const b = (batch.get(r.custom_id) ?? '').trim();
                expect(b.length).toBeGreaterThan(0);
                const exact = s.slice(0, 120) === b.slice(0, 120);
                if (exact) {
                    firstLineMatches++;
                }
                writeFileSync(join(outDir, `${r.custom_id}.sync.md`), s);
                writeFileSync(join(outDir, `${r.custom_id}.batch.md`), b);
                perPage.push({
                    custom_id: r.custom_id,
                    sync_chars: s.length,
                    batch_chars: b.length,
                    first120_exact: exact,
                    identical: s === b,
                    batch_tokens: batchTokens.get(r.custom_id),
                });
            }
            const summary = {
                model: MODEL,
                pages: requests.length,
                sync_ms: syncMs,
                batch_ms: batchMs,
                batch_job_id: job.id,
                first120_exact_matches: `${firstLineMatches}/${requests.length}`,
                per_page: perPage,
            };
            writeFileSync(join(outDir, 'summary.json'), JSON.stringify(summary, null, 2));
            // eslint-disable-next-line no-console
            console.log(`[batch-vs-sync] ${JSON.stringify(summary)}`);
            // Report-only: batch and sync can differ slightly (sampling); we assert both produced output.
            expect(firstLineMatches).toBeGreaterThanOrEqual(0);
        },
        TIMEOUT_MS,
    );
});
