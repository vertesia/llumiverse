/**
 * Vertex batch SCALE test — batch-only turnaround vs page count, using GCS-referenced
 * images (the real PdfToMd path: images already in GCS → referenced by gs:// URI, so
 * the JSONL stays ~1KB/line at any scale). Sync latency is INFERRED from the measured
 * per-page rate (no need to actually run hours of sync).
 *
 * Answers: "at what scale does batch become faster than sync, and by how much?"
 *
 * Gated:
 *   GOOGLE_PROJECT_ID, GOOGLE_REGION (default us-central1)
 *   BATCH_BUCKET        gs bucket/prefix the caller can write to
 *   BATCH_TEST_PDF      source PDF (pages are rendered once, uploaded once, then cycled)
 *   BATCH_SCALE_PAGES   number of batch requests to submit (default 1000)
 *   BATCH_TEST_MODEL    default gemini-2.5-flash-lite
 *   SYNC_S_PER_PAGE     measured sync rate for inference (default 3.5)
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
    type BatchInferenceRequestItem,
    type ExecutionOptions,
    type PromptSegment,
    URLDataSource,
} from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { VertexAIDriver } from '../src/index.js';

const PROJECT = process.env.GOOGLE_PROJECT_ID;
const REGION = process.env.GOOGLE_REGION ?? 'us-central1';
const BUCKET = process.env.BATCH_BUCKET;
const PDF = process.env.BATCH_TEST_PDF;
const MODEL = process.env.BATCH_TEST_MODEL ?? 'gemini-2.5-flash-lite';
const SCALE = Number(process.env.BATCH_SCALE_PAGES ?? '1000');
const SYNC_S_PER_PAGE = Number(process.env.SYNC_S_PER_PAGE ?? '3.5');
const POLL_MS = 30_000;
const TIMEOUT_MS = 4 * 60 * 60 * 1000; // 4h

const gated = Boolean(PROJECT && BUCKET && PDF && existsSync(PDF));
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function parseBucket(spec: string): { bucket: string; prefix: string } {
    const s = spec.replace(/^gs:\/\//, '');
    const slash = s.indexOf('/');
    return slash === -1
        ? { bucket: s, prefix: '' }
        : { bucket: s.slice(0, slash), prefix: s.slice(slash + 1).replace(/\/+$/, '') };
}

describe.skipIf(!gated)('Vertex batch scale — turnaround vs page count', () => {
    it(
        `batch-only at ${SCALE} pages (GCS-referenced images)`,
        async () => {
            // 1. render all pages once, 2. upload to GCS once (referenced, not inlined)
            const dir = mkdtempSync(join(tmpdir(), 'scaleimg-'));
            execFileSync('pdftoppm', ['-r', '150', '-jpeg', PDF as string, join(dir, 'p')]);
            const files = readdirSync(dir)
                .filter((f) => f.endsWith('.jpg'))
                .sort();
            expect(files.length).toBeGreaterThan(0);

            const { bucket, prefix } = parseBucket(BUCKET as string);
            const imgPrefix = `${prefix ? `${prefix}/` : ''}scale-imgs`;
            execFileSync('gcloud', [
                'storage',
                'cp',
                ...files.map((f) => join(dir, f)),
                `gs://${bucket}/${imgPrefix}/`,
                '--project',
                PROJECT as string,
            ]);
            const gsUris = files.map((f) => `gs://${bucket}/${imgPrefix}/${f}`);

            // 3. build SCALE requests cycling the GCS images
            const PROMPT =
                'Convert this document page image to clean GitHub-flavored Markdown. Output only the markdown, with no preamble.';
            const requests: BatchInferenceRequestItem[] = Array.from({ length: SCALE }, (_, i) => ({
                custom_id: `page-${i + 1}`,
                segments: [
                    { role: 'system', content: PROMPT },
                    {
                        role: 'user',
                        content: '',
                        files: [
                            new URLDataSource(`img-${i % gsUris.length}.jpg`, 'image/jpeg', gsUris[i % gsUris.length]),
                        ],
                    },
                ] as PromptSegment[],
                options: { model: MODEL, model_options: { max_tokens: 4096, temperature: 0 } } as ExecutionOptions,
            }));

            const driver = new VertexAIDriver({ project: PROJECT as string, region: REGION, batch_bucket: BUCKET });

            // 4. batch-only: measure submit → succeeded → results turnaround
            const t0 = Date.now();
            const job = await driver.startBatchInference(requests, { name: `scale-${SCALE}` });
            // eslint-disable-next-line no-console
            console.log(`[scale ${SCALE}] submitted ${job.id} status=${job.status}`);
            let status = job;
            while (status.status === 'queued' || status.status === 'running') {
                await sleep(POLL_MS);
                status = await driver.getBatchInferenceJob(job.id);
                // eslint-disable-next-line no-console
                console.log(`[scale ${SCALE}] ${new Date().toISOString()} status=${status.status}`);
            }
            const batchMs = Date.now() - t0;
            expect(status.status).toBe('succeeded');

            const results = await driver.getBatchInferenceResults(job.id);
            const nonEmpty = results.filter((r) => (r.result?.length ?? 0) > 0).length;
            const syncMs = SCALE * SYNC_S_PER_PAGE * 1000;
            const summary = {
                model: MODEL,
                pages: SCALE,
                batch_ms: batchMs,
                batch_min: +(batchMs / 60000).toFixed(1),
                inferred_sync_min: +(syncMs / 60000).toFixed(1),
                speedup_vs_sync: +(syncMs / batchMs).toFixed(2),
                results: results.length,
                non_empty: nonEmpty,
                job: job.id,
            };
            const outDir = process.env.BATCH_OUT_DIR ?? join(tmpdir(), 'batch-scale');
            mkdirSync(outDir, { recursive: true });
            writeFileSync(join(outDir, `scale-${SCALE}.json`), JSON.stringify(summary, null, 2));
            // eslint-disable-next-line no-console
            console.log(`[scale-result] ${JSON.stringify(summary)}`);
            expect(results.length).toBe(SCALE);
        },
        TIMEOUT_MS,
    );
});
