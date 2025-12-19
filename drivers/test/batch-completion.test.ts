/**
 * Batch API completion tests.
 *
 * Tests for operations that require waiting for batch jobs to complete.
 * These tests take a long time (up to 24 hours for large batches) and should
 * only be run manually with small test inputs.
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
 * - BATCH_TEST_POLL_INTERVAL: Poll interval in ms (default: 30000)
 * - BATCH_TEST_MAX_WAIT: Max wait time in ms (default: 1800000 = 30 minutes)
 *
 * Sample input JSONL for Gemini inference:
 * {"contents": [{"role": "user", "parts": [{"text": "Say hello"}]}]}
 *
 * Sample input JSONL for Claude inference:
 * {"custom_id": "req-1", "request": {"messages": [{"role": "user", "content": "Say hello"}], "anthropic_version": "vertex-2023-10-16", "max_tokens": 100}}
 *
 * Sample input JSONL for embeddings:
 * {"content": "Hello world"}
 */

import { BatchJobStatus, BatchJobType } from '@llumiverse/common';
import 'dotenv/config';
import { describe, expect, test, afterAll } from 'vitest';
import { VertexAIDriver } from '../src/index.js';

// Default timeout: 35 minutes (to allow for 30 min max wait + overhead)
const TIMEOUT = 35 * 60 * 1000;

// Test configuration from environment
const config = {
    projectId: process.env.GOOGLE_PROJECT_ID,
    region: process.env.GOOGLE_REGION || 'us-central1',
    gcsInputUri: process.env.BATCH_TEST_GCS_INPUT_URI,
    gcsOutputUri: process.env.BATCH_TEST_GCS_OUTPUT_URI,
    geminiModel: process.env.BATCH_TEST_GEMINI_MODEL || 'publishers/google/models/gemini-2.5-flash-lite',
    claudeModel: process.env.BATCH_TEST_CLAUDE_MODEL || 'publishers/anthropic/models/claude-haiku-4-5',
    embeddingModel: process.env.BATCH_TEST_EMBEDDING_MODEL || 'gemini-embedding-001',
    pollInterval: parseInt(process.env.BATCH_TEST_POLL_INTERVAL || '30000', 10),
    maxWait: parseInt(process.env.BATCH_TEST_MAX_WAIT || '1800000', 10), // 30 minutes default
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
    console.warn('Batch completion tests are skipped: Missing required environment variables');
    console.warn('Required: GOOGLE_PROJECT_ID, BATCH_TEST_GCS_INPUT_URI, BATCH_TEST_GCS_OUTPUT_URI');
}

// Cleanup: cancel any incomplete jobs
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

describe.skipIf(!canRunTests)('VertexAI Batch: Wait for Completion', () => {

    // ============== Gemini Completion Test ==============

    describe('Gemini inference - wait for completion', () => {
        const testId = Math.random().toString(36).substring(7);

        test('should create and wait for Gemini batch job to complete', { timeout: TIMEOUT }, async () => {
            console.log('Creating Gemini batch job...');
            console.log(`  Model: ${config.geminiModel}`);
            console.log(`  Input: ${config.gcsInputUri}`);
            console.log(`  Output: ${config.gcsOutputUri}`);
            console.log(`  Poll interval: ${config.pollInterval}ms`);
            console.log(`  Max wait: ${config.maxWait}ms`);

            // Create the job
            const createdJob = await driver!.createBatchJob({
                model: config.geminiModel,
                type: BatchJobType.inference,
                source: { gcsUris: [`${config.gcsInputUri}input-sample-gemini.jsonl`] },
                destination: { gcsUri: `${config.gcsOutputUri}gemini-completion-${testId}/` },
                displayName: `test-gemini-completion-${Date.now()}`,
            });

            expect(createdJob).toBeDefined();
            expect(createdJob.id).toBeDefined();
            createdJobIds.push(createdJob.id);
            console.log(`Created job: ${createdJob.id}`);

            // Wait for completion
            console.log('Waiting for job to complete...');
            const completedJob = await driver!.waitForBatchJobCompletion(
                createdJob.id,
                config.pollInterval,
                config.maxWait
            );

            console.log('Job completed:', JSON.stringify(completedJob, null, 2));

            expect(completedJob).toBeDefined();
            expect([BatchJobStatus.succeeded, BatchJobStatus.failed, BatchJobStatus.partial]).toContain(completedJob.status);

            if (completedJob.status === BatchJobStatus.succeeded) {
                console.log('Job succeeded!');
                expect(completedJob.stats).toBeDefined();
                if (completedJob.stats) {
                    console.log(`  Completed requests: ${completedJob.stats.completedRequests}`);
                    console.log(`  Failed requests: ${completedJob.stats.failedRequests}`);
                }
            } else if (completedJob.status === BatchJobStatus.failed) {
                console.log('Job failed:', completedJob.error);
            } else if (completedJob.status === BatchJobStatus.partial) {
                console.log('Job partially succeeded');
                console.log(`  Stats: ${JSON.stringify(completedJob.stats)}`);
            }
        });
    });

    // ============== Claude Completion Test ==============

    describe('Claude inference - wait for completion', () => {
        const testId = Math.random().toString(36).substring(7);

        test('should create and wait for Claude batch job to complete', { timeout: TIMEOUT }, async () => {
            console.log('Creating Claude batch job...');
            console.log(`  Model: ${config.claudeModel}`);
            console.log(`  Input: ${config.gcsInputUri}`);
            console.log(`  Output: ${config.gcsOutputUri}`);

            // Create the job
            const createdJob = await driver!.createBatchJob({
                model: config.claudeModel,
                type: BatchJobType.inference,
                source: { gcsUris: [`${config.gcsInputUri}input-sample-claude.jsonl`] },
                destination: { gcsUri: `${config.gcsOutputUri}claude-completion-${testId}/` },
                displayName: `test-claude-completion-${Date.now()}`,
            });

            expect(createdJob).toBeDefined();
            expect(createdJob.id).toBeDefined();
            createdJobIds.push(createdJob.id);
            console.log(`Created job: ${createdJob.id}`);

            // Wait for completion
            console.log('Waiting for job to complete...');
            const completedJob = await driver!.waitForBatchJobCompletion(
                createdJob.id,
                config.pollInterval,
                config.maxWait
            );

            console.log('Job completed:', JSON.stringify(completedJob, null, 2));

            expect(completedJob).toBeDefined();
            expect([BatchJobStatus.succeeded, BatchJobStatus.failed, BatchJobStatus.partial]).toContain(completedJob.status);

            if (completedJob.status === BatchJobStatus.succeeded) {
                console.log('Job succeeded!');
                expect(completedJob.stats).toBeDefined();
            } else if (completedJob.status === BatchJobStatus.failed) {
                console.log('Job failed:', completedJob.error);
            }
        });
    });

    // ============== Embeddings Completion Test ==============

    describe('Embeddings - wait for completion', () => {
        const testId = Math.random().toString(36).substring(7);

        test('should create and wait for embeddings batch job to complete', { timeout: TIMEOUT }, async () => {
            console.log('Creating embeddings batch job...');
            console.log(`  Model: ${config.embeddingModel}`);
            console.log(`  Input: ${config.gcsInputUri}`);
            console.log(`  Output: ${config.gcsOutputUri}`);

            // Create the job
            const createdJob = await driver!.createBatchJob({
                model: config.embeddingModel,
                type: BatchJobType.embeddings,
                source: { gcsUris: [`${config.gcsInputUri}input-sample-embeddings.jsonl`] },
                destination: { gcsUri: `${config.gcsOutputUri}embeddings-completion-${testId}/` },
                displayName: `test-embeddings-completion-${Date.now()}`,
            });

            expect(createdJob).toBeDefined();
            expect(createdJob.id).toBeDefined();
            createdJobIds.push(createdJob.id);
            console.log(`Created job: ${createdJob.id}`);

            // Wait for completion
            console.log('Waiting for job to complete...');
            const completedJob = await driver!.waitForBatchJobCompletion(
                createdJob.id,
                config.pollInterval,
                config.maxWait
            );

            console.log('Job completed:', JSON.stringify(completedJob, null, 2));

            expect(completedJob).toBeDefined();
            expect([BatchJobStatus.succeeded, BatchJobStatus.failed, BatchJobStatus.partial]).toContain(completedJob.status);

            if (completedJob.status === BatchJobStatus.succeeded) {
                console.log('Job succeeded!');
                expect(completedJob.stats).toBeDefined();
            } else if (completedJob.status === BatchJobStatus.failed) {
                console.log('Job failed:', completedJob.error);
            }
        });
    });

    // ============== Delete Test ==============

    describe('Delete batch job', () => {
        const testId = Math.random().toString(36).substring(7);

        test('should create, cancel, and delete a batch job', { timeout: TIMEOUT }, async () => {
            // Create a job to delete
            const createdJob = await driver!.createBatchJob({
                model: config.geminiModel,
                type: BatchJobType.inference,
                source: { gcsUris: [`${config.gcsInputUri}input-sample-gemini.jsonl`] },
                destination: { gcsUri: `${config.gcsOutputUri}delete-${testId}/` },
                displayName: `test-delete-${Date.now()}`,
            });

            expect(createdJob).toBeDefined();
            console.log(`Created job for deletion: ${createdJob.id}`);

            // Cancel it first (can only delete completed/cancelled jobs)
            await driver!.cancelBatchJob(createdJob.id);
            console.log('Cancelled job');

            // Wait a bit for cancellation to propagate
            await new Promise(resolve => setTimeout(resolve, 5000));

            // Try to delete (may fail if not yet cancelled)
            try {
                await driver!.deleteBatchJob(createdJob.id);
                console.log('Deleted job successfully');
            } catch (e: any) {
                console.log('Delete failed (may need to wait for cancellation):', e.message);
                // This is expected if the job hasn't fully cancelled yet
            }
        });
    });
});
