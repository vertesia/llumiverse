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

import { BatchJobStatus, BatchJobType } from '@llumiverse/common';
import 'dotenv/config';
import { describe, expect, test, beforeAll, afterAll } from 'vitest';
import { VertexAIDriver } from '../src/index.js';

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

let driver: VertexAIDriver | undefined;
const createdJobIds: string[] = [];

if (canRunTests) {
    driver = new VertexAIDriver({
        project: config.projectId!,
        region: config.region,
    });
} else {
    console.warn('Batch tests are skipped: Missing required environment variables');
    console.warn('Required: GOOGLE_PROJECT_ID, BATCH_TEST_GCS_INPUT_URI, BATCH_TEST_GCS_OUTPUT_URI');
}

// Cleanup: cancel any jobs we created during testing
afterAll(async () => {
    if (!driver) return;

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
});

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
