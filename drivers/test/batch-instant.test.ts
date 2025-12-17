/**
 * Batch API instant operations tests.
 *
 * Tests for operations that return quickly: create, list, get, cancel, delete.
 * These tests create actual batch jobs and incur costs - run manually when needed.
 *
 * Required environment variables:
 * - GOOGLE_PROJECT_ID: GCP project ID
 * - GOOGLE_REGION: GCP region (e.g., us-central1)
 * - BATCH_TEST_GCS_INPUT_URI: GCS URI for input JSONL file (e.g., gs://bucket/input.jsonl)
 * - BATCH_TEST_GCS_OUTPUT_URI: GCS URI prefix for output (e.g., gs://bucket/output/)
 *
 * Optional:
 * - BATCH_TEST_GEMINI_MODEL: Gemini model to test (default: gemini-2.0-flash)
 * - BATCH_TEST_CLAUDE_MODEL: Claude model to test (default: claude-3-5-haiku)
 * - BATCH_TEST_EMBEDDING_MODEL: Embedding model to test (default: gemini-embedding-001)
 */

import { BatchJobStatus, BatchJobType, PromptRole } from '@llumiverse/common';
import 'dotenv/config';
import { describe, expect, test, beforeAll, afterAll } from 'vitest';
import { VertexAIDriver } from '../src/index.js';
import { uploadBatchInput } from '../src/vertexai/batch/gcs-helpers.js';
import { formatGeminiRequest, formatClaudeRequest, formatEmbeddingsRequest } from '../src/vertexai/batch/formatters.js';

const TIMEOUT = 60 * 1000; // 60 seconds for batch operations

// Test configuration from environment
const config = {
    projectId: process.env.GOOGLE_PROJECT_ID,
    region: process.env.GOOGLE_REGION || 'us-central1',
    gcsInputUri: process.env.BATCH_TEST_GCS_INPUT_URI,
    gcsOutputUri: process.env.BATCH_TEST_GCS_OUTPUT_URI,
    geminiModel: process.env.BATCH_TEST_GEMINI_MODEL || 'publishers/google/models/gemini-2.5-flash-lite',
    claudeModel: process.env.BATCH_TEST_CLAUDE_MODEL || 'publishers/anthropic/models/claude-haiku-4-5',
    embeddingModel: process.env.BATCH_TEST_EMBEDDING_MODEL || 'gemini-embedding-001',
};

// Skip all tests if required env vars are missing
const canRunTests = config.projectId && config.gcsInputUri && config.gcsOutputUri;

// Parse bucket from output URI for manual upload tests
const batchBucket = config.gcsOutputUri?.match(/^gs:\/\/([^/]+)/)?.[1];

let driver: VertexAIDriver | undefined;
const createdJobIds: string[] = [];
const testFileUris: string[] = []; // Track uploaded test files for cleanup

if (canRunTests) {
    driver = new VertexAIDriver({
        project: config.projectId!,
        region: config.region,
    });
}

if (!canRunTests) {
    console.warn('Batch tests are skipped: Missing required environment variables');
    console.warn('Required: GOOGLE_PROJECT_ID, BATCH_TEST_GCS_INPUT_URI, BATCH_TEST_GCS_OUTPUT_URI');
}

// Cleanup: cancel jobs and delete test files we created during testing
afterAll(async () => {
    if (!driver) return;

    // Cancel any pending/running jobs
    for (const jobId of createdJobIds) {
        try {
            const job = await driver.getBatchJob(jobId);
            if (job.status === BatchJobStatus.pending || job.status === BatchJobStatus.running) {
                console.log(`Cleaning up: cancelling job ${jobId}`);
                await driver.cancelBatchJob(jobId);
            }
        } catch (e) {
            console.warn(`Failed to cleanup job ${jobId}:`, e);
        }
    }

    // Delete test files from GCS
    if (batchBucket) {
        const { Storage } = await import('@google-cloud/storage');
        const authClient = await driver.googleAuth.getClient();
        const storage = new Storage({ authClient: authClient as any });

        for (const fileUri of testFileUris) {
            try {
                const { bucket, path } = parseGcsUri(fileUri);
                console.log(`Cleaning up: deleting test file ${fileUri}`);
                await storage.bucket(bucket).file(path).delete();
            } catch (e) {
                console.warn(`Failed to cleanup test file ${fileUri}:`, e);
            }
        }
    }
});

/**
 * Parses a GCS URI into bucket and path components.
 */
function parseGcsUri(uri: string): { bucket: string; path: string } {
    const match = uri.match(/^gs:\/\/([^/]+)\/(.+)$/);
    if (!match) {
        throw new Error(`Invalid GCS URI: ${uri}`);
    }
    return { bucket: match[1], path: match[2] };
}

describe.skipIf(!canRunTests)('VertexAI Batch: Instant Operations', () => {

    // ============== List Operations ==============

    describe('List batch jobs', () => {
        test('should list batch jobs', { timeout: TIMEOUT }, async () => {
            const result = await driver!.listBatchJobs({ pageSize: 10 });
            expect(result).toBeDefined();
            expect(result.jobs).toBeDefined();
            expect(Array.isArray(result.jobs)).toBe(true);
            console.log(`Found ${result.jobs.length} batch jobs`);
            if (result.jobs.length > 0) {
                console.log('First job:', JSON.stringify(result.jobs[0], null, 2));
            }
        });

        test('should list batch jobs with pagination', { timeout: TIMEOUT }, async () => {
            const result = await driver!.listBatchJobs({ pageSize: 2 });
            expect(result).toBeDefined();
            expect(result.jobs.length).toBeLessThanOrEqual(2);
            // Note: nextPageToken may be undefined if fewer jobs exist
        });
    });

    // ============== Gemini Batch Operations ==============

    describe('Gemini inference batch', () => {
        let geminiJobId: string | undefined;

        test('should create a Gemini inference batch job', { timeout: TIMEOUT }, async () => {
            const job = await driver!.createBatchJob({
                model: config.geminiModel,
                type: BatchJobType.inference,
                source: { gcsUris: [config.gcsInputUri!] },
                destination: { gcsUri: config.gcsOutputUri },
                displayName: `test-gemini-${Date.now()}`,
            });

            expect(job).toBeDefined();
            expect(job.id).toBeDefined();
            expect(job.status).toBe(BatchJobStatus.pending);
            expect(job.type).toBe(BatchJobType.inference);
            expect(job.provider).toBe('vertexai');

            geminiJobId = job.id;
            createdJobIds.push(job.id);
            console.log('Created Gemini batch job:', JSON.stringify(job, null, 2));
        });

        test('should get Gemini batch job status', { timeout: TIMEOUT }, async () => {
            if (!geminiJobId) {
                console.warn('Skipping: No Gemini job created');
                return;
            }

            const job = await driver!.getBatchJob(geminiJobId);
            expect(job).toBeDefined();
            expect(job.id).toBe(geminiJobId);
            expect([BatchJobStatus.pending, BatchJobStatus.running]).toContain(job.status);
            console.log('Gemini job status:', job.status);
        });

        test('should cancel Gemini batch job', { timeout: TIMEOUT }, async () => {
            if (!geminiJobId) {
                console.warn('Skipping: No Gemini job created');
                return;
            }

            const job = await driver!.cancelBatchJob(geminiJobId);
            expect(job).toBeDefined();
            // Job may already be cancelled or in cancelling state
            expect([BatchJobStatus.cancelled, BatchJobStatus.pending, BatchJobStatus.running]).toContain(job.status);
            console.log('Gemini job after cancel:', job.status);
        });
    });

    // ============== Claude Batch Operations ==============

    describe('Claude inference batch', () => {
        let claudeJobId: string | undefined;

        test('should create a Claude inference batch job', { timeout: TIMEOUT }, async () => {
            const job = await driver!.createBatchJob({
                model: config.claudeModel,
                type: BatchJobType.inference,
                source: { gcsUris: [config.gcsInputUri!] },
                destination: { gcsUri: config.gcsOutputUri },
                displayName: `test-claude-${Date.now()}`,
            });

            expect(job).toBeDefined();
            expect(job.id).toBeDefined();
            expect(job.status).toBe(BatchJobStatus.pending);
            expect(job.type).toBe(BatchJobType.inference);
            expect(job.provider).toBe('vertexai');

            claudeJobId = job.id;
            createdJobIds.push(job.id);
            console.log('Created Claude batch job:', JSON.stringify(job, null, 2));
        });

        test('should get Claude batch job status', { timeout: TIMEOUT }, async () => {
            if (!claudeJobId) {
                console.warn('Skipping: No Claude job created');
                return;
            }

            const job = await driver!.getBatchJob(claudeJobId);
            expect(job).toBeDefined();
            expect(job.id).toBe(claudeJobId);
            expect([BatchJobStatus.pending, BatchJobStatus.running]).toContain(job.status);
            console.log('Claude job status:', job.status);
        });

        test('should cancel Claude batch job', { timeout: TIMEOUT }, async () => {
            if (!claudeJobId) {
                console.warn('Skipping: No Claude job created');
                return;
            }

            const job = await driver!.cancelBatchJob(claudeJobId);
            expect(job).toBeDefined();
            expect([BatchJobStatus.cancelled, BatchJobStatus.pending, BatchJobStatus.running]).toContain(job.status);
            console.log('Claude job after cancel:', job.status);
        });
    });

    // ============== Embeddings Batch Operations ==============

    describe('Embeddings batch', () => {
        let embeddingsJobId: string | undefined;

        test('should create an embeddings batch job', { timeout: TIMEOUT }, async () => {
            const job = await driver!.createBatchJob({
                model: config.embeddingModel,
                type: BatchJobType.embeddings,
                source: { gcsUris: [config.gcsInputUri!] },
                destination: { gcsUri: config.gcsOutputUri },
                displayName: `test-embeddings-${Date.now()}`,
            });

            expect(job).toBeDefined();
            expect(job.id).toBeDefined();
            expect(job.status).toBe(BatchJobStatus.pending);
            expect(job.type).toBe(BatchJobType.embeddings);
            expect(job.provider).toBe('vertexai');

            embeddingsJobId = job.id;
            createdJobIds.push(job.id);
            console.log('Created embeddings batch job:', JSON.stringify(job, null, 2));
        });

        test('should get embeddings batch job status', { timeout: TIMEOUT }, async () => {
            if (!embeddingsJobId) {
                console.warn('Skipping: No embeddings job created');
                return;
            }

            const job = await driver!.getBatchJob(embeddingsJobId);
            expect(job).toBeDefined();
            expect(job.id).toBe(embeddingsJobId);
            expect([BatchJobStatus.pending, BatchJobStatus.running]).toContain(job.status);
            console.log('Embeddings job status:', job.status);
        });

        test('should cancel embeddings batch job', { timeout: TIMEOUT }, async () => {
            if (!embeddingsJobId) {
                console.warn('Skipping: No embeddings job created');
                return;
            }

            const job = await driver!.cancelBatchJob(embeddingsJobId);
            expect(job).toBeDefined();
            expect([BatchJobStatus.cancelled, BatchJobStatus.pending, BatchJobStatus.running]).toContain(job.status);
            console.log('Embeddings job after cancel:', job.status);
        });
    });
});

// ============== Manual JSONL Upload Tests ==============
// These tests manually create JSONL files and upload them to GCS

describe.skipIf(!canRunTests || !batchBucket)('VertexAI Batch: Manual JSONL Upload', () => {

    describe('Gemini with manual JSONL upload', () => {
        let geminiJobId: string | undefined;
        let geminiTestId: string;
        let geminiInputUri: string;

        test('should create Gemini batch job with manually uploaded JSONL', { timeout: TIMEOUT }, async () => {
            geminiTestId = Math.random().toString(36).substring(7);

            // Create message sets (same format as regular inference)
            const messageSets = [
                [{ role: PromptRole.user, content: 'What is 2+2?' }],
                [{ role: PromptRole.user, content: 'What is the capital of France?' }],
            ];

            // Format to JSONL
            const lines = messageSets.map(msgs => formatGeminiRequest(msgs));

            // Upload to GCS
            geminiInputUri = await uploadBatchInput(
                driver!,
                lines,
                batchBucket!,
                `batch-input-gemini-${geminiTestId}.jsonl`
            );
            testFileUris.push(geminiInputUri);

            // Create batch job
            const job = await driver!.createBatchJob({
                model: config.geminiModel,
                type: BatchJobType.inference,
                source: { gcsUris: [geminiInputUri] },
                destination: { gcsUri: `gs://${batchBucket}/batch-output-gemini-${geminiTestId}/` },
                displayName: `test-gemini-manual-${Date.now()}`,
            });

            expect(job).toBeDefined();
            expect(job.id).toBeDefined();
            expect(job.status).toBe(BatchJobStatus.pending);
            expect(job.source?.gcsUris?.[0]).toBe(geminiInputUri);

            geminiJobId = job.id;
            createdJobIds.push(job.id);
            console.log('Created Gemini batch job with manual upload:', JSON.stringify(job, null, 2));
        });

        test('should cancel Gemini batch job', { timeout: TIMEOUT }, async () => {
            if (!geminiJobId) {
                console.warn('Skipping: No Gemini job created');
                return;
            }

            const job = await driver!.cancelBatchJob(geminiJobId);
            expect(job).toBeDefined();
            expect([BatchJobStatus.cancelled, BatchJobStatus.pending, BatchJobStatus.running]).toContain(job.status);
            console.log('Gemini job after cancel:', job.status);
        });
    });

    describe('Claude with manual JSONL upload', () => {
        let claudeJobId: string | undefined;
        let claudeTestId: string;
        let claudeInputUri: string;

        test('should create Claude batch job with manually uploaded JSONL', { timeout: TIMEOUT }, async () => {
            claudeTestId = Math.random().toString(36).substring(7);

            // Create message sets (same format as regular inference)
            const messageSets = [
                [{ role: PromptRole.user, content: 'Hello Claude!' }],
            ];

            // Format to JSONL (Claude requires custom_id for each request)
            const lines = messageSets.map((msgs, idx) => formatClaudeRequest(msgs, `req-${idx + 1}`));

            // Upload to GCS
            claudeInputUri = await uploadBatchInput(
                driver!,
                lines,
                batchBucket!,
                `batch-input-claude-${claudeTestId}.jsonl`
            );
            testFileUris.push(claudeInputUri);

            // Create batch job
            const job = await driver!.createBatchJob({
                model: config.claudeModel,
                type: BatchJobType.inference,
                source: { gcsUris: [claudeInputUri] },
                destination: { gcsUri: `gs://${batchBucket}/batch-output-claude-${claudeTestId}/` },
                displayName: `test-claude-manual-${Date.now()}`,
            });

            expect(job).toBeDefined();
            expect(job.id).toBeDefined();
            expect(job.status).toBe(BatchJobStatus.pending);
            expect(job.source?.gcsUris?.[0]).toBe(claudeInputUri);

            claudeJobId = job.id;
            createdJobIds.push(job.id);
            console.log('Created Claude batch job with manual upload:', JSON.stringify(job, null, 2));
        });

        test('should cancel Claude batch job', { timeout: TIMEOUT }, async () => {
            if (!claudeJobId) {
                console.warn('Skipping: No Claude job created');
                return;
            }

            const job = await driver!.cancelBatchJob(claudeJobId);
            expect(job).toBeDefined();
            expect([BatchJobStatus.cancelled, BatchJobStatus.pending, BatchJobStatus.running]).toContain(job.status);
            console.log('Claude job after cancel:', job.status);
        });
    });

    describe('Embeddings with manual JSONL upload', () => {
        let embeddingsJobId: string | undefined;
        let embeddingsTestId: string;
        let embeddingsInputUri: string;

        test('should create embeddings batch job with manually uploaded JSONL', { timeout: TIMEOUT }, async () => {
            embeddingsTestId = Math.random().toString(36).substring(7);

            // Create message sets (same format as regular inference)
            const messageSets = [
                [{ role: PromptRole.user, content: 'Hello world' }],
                [{ role: PromptRole.user, content: 'Goodbye world' }],
            ];

            // Format to JSONL for embeddings
            const lines = messageSets.map(msgs => formatEmbeddingsRequest(msgs));

            // Upload to GCS
            embeddingsInputUri = await uploadBatchInput(
                driver!,
                lines,
                batchBucket!,
                `batch-input-embeddings-${embeddingsTestId}.jsonl`
            );
            testFileUris.push(embeddingsInputUri);

            // Create batch job
            const job = await driver!.createBatchJob({
                model: config.embeddingModel,
                type: BatchJobType.embeddings,
                source: { gcsUris: [embeddingsInputUri] },
                destination: { gcsUri: `gs://${batchBucket}/batch-output-embeddings-${embeddingsTestId}/` },
                displayName: `test-embeddings-manual-${Date.now()}`,
            });

            expect(job).toBeDefined();
            expect(job.id).toBeDefined();
            expect(job.status).toBe(BatchJobStatus.pending);
            expect(job.source?.gcsUris?.[0]).toBe(embeddingsInputUri);

            embeddingsJobId = job.id;
            createdJobIds.push(job.id);
            console.log('Created embeddings batch job with manual upload:', JSON.stringify(job, null, 2));
        });

        test('should cancel embeddings batch job', { timeout: TIMEOUT }, async () => {
            if (!embeddingsJobId) {
                console.warn('Skipping: No embeddings job created');
                return;
            }

            const job = await driver!.cancelBatchJob(embeddingsJobId);
            expect(job).toBeDefined();
            expect([BatchJobStatus.cancelled, BatchJobStatus.pending, BatchJobStatus.running]).toContain(job.status);
            console.log('Embeddings job after cancel:', job.status);
        });
    });
});
