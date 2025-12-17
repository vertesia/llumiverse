# Batch API Implementation Guide

> **Note**: This file is temporary and should be deleted once Bedrock and OpenAI batch implementations are complete.

## Overview

This document provides detailed information for implementing batch support in additional providers (Bedrock, OpenAI). The VertexAI implementation serves as the reference.

## Architecture

### Common Types (already implemented)

Location: `common/src/batch.ts`

```typescript
// Core types - use these, don't create provider-specific versions
export enum BatchJobStatus {
    pending = "pending",
    running = "running",
    succeeded = "succeeded",
    failed = "failed",
    cancelled = "cancelled",
    partial = "partial"  // Some requests failed
}

export enum BatchJobType {
    inference = "inference",
    embeddings = "embeddings"
}

export interface BatchJob {
    id: string;                    // Encoded: "provider:providerJobId"
    displayName?: string;
    status: BatchJobStatus;
    type: BatchJobType;
    model: string;
    source: BatchJobSource;
    destination?: BatchJobDestination;
    createdAt?: Date;
    completedAt?: Date;
    error?: BatchJobError;
    stats?: BatchJobStats;
    provider: string;              // "vertexai", "bedrock", "openai"
    providerJobId?: string;        // Original provider job ID
}

export interface CreateBatchJobOptions {
    model: string;
    type: BatchJobType;
    source: BatchJobSource;        // { gcsUris?, s3Uris?, bigqueryUri? }
    destination: BatchJobDestination;
    displayName?: string;
    modelOptions?: Record<string, unknown>;
}
```

### File Structure Pattern

```
drivers/src/{provider}/
├── batch/
│   ├── index.ts           # {Provider}BatchClient class
│   ├── types.ts           # Provider-specific types & state mapping
│   └── {model}-batch.ts   # Model-specific implementations (if needed)
└── index.ts               # Add getBatchClient() method
```

---

## OpenAI Batch Implementation

### API Documentation

- https://platform.openai.com/docs/guides/batch
- https://platform.openai.com/docs/api-reference/batch

### Key Details

**SDK**: Use `openai` npm package

```typescript
import OpenAI from 'openai';

// Create batch
const batch = await openai.batches.create({
    input_file_id: "file-abc123",  // Must upload file first
    endpoint: "/v1/chat/completions",
    completion_window: "24h"
});

// Get batch
const batch = await openai.batches.retrieve("batch_abc123");

// List batches
const batches = await openai.batches.list({ limit: 10 });

// Cancel batch
await openai.batches.cancel("batch_abc123");
```

**Job States to Map**:

| OpenAI State | BatchJobStatus |
|--------------|----------------|
| validating | pending |
| in_progress | running |
| completed | succeeded |
| failed | failed |
| cancelled | cancelled |
| expired | failed |
| finalizing | running |
| cancelling | cancelled |

**Input Format** (JSONL):

```json
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}}
```

**File Handling**:
- OpenAI requires uploading files first via Files API
- Input: `openai.files.create({ file, purpose: "batch" })`
- Output: Download via `openai.files.content(output_file_id)`

**Embeddings**:
- Endpoint: `/v1/embeddings`
- Same batch mechanism, different body format

### Implementation Notes

1. Need to handle file upload/download (OpenAI doesn't use S3 directly)
2. Consider adding helper methods for file operations
3. `completion_window` is always "24h" currently

---

## AWS Bedrock Batch Implementation

### API Documentation

- https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html
- https://docs.aws.amazon.com/bedrock/latest/APIReference/API_CreateModelInvocationJob.html

### Key Details

**SDK**: Use `@aws-sdk/client-bedrock`

```typescript
import {
    BedrockClient,
    CreateModelInvocationJobCommand,
    GetModelInvocationJobCommand,
    ListModelInvocationJobsCommand,
    StopModelInvocationJobCommand
} from "@aws-sdk/client-bedrock";

// Create batch job
const response = await client.send(new CreateModelInvocationJobCommand({
    jobName: "my-batch-job",
    modelId: "anthropic.claude-3-haiku-20240307-v1:0",
    roleArn: "arn:aws:iam::123456789:role/BedrockBatchRole",
    inputDataConfig: {
        s3InputDataConfig: {
            s3Uri: "s3://bucket/input/",
            s3InputFormat: "JSONL"
        }
    },
    outputDataConfig: {
        s3OutputDataConfig: {
            s3Uri: "s3://bucket/output/"
        }
    }
}));

// Get job
const job = await client.send(new GetModelInvocationJobCommand({
    jobIdentifier: "job-id"
}));

// List jobs
const jobs = await client.send(new ListModelInvocationJobsCommand({
    maxResults: 10
}));

// Stop job
await client.send(new StopModelInvocationJobCommand({
    jobIdentifier: "job-id"
}));
```

**Job States to Map**:

| Bedrock State | BatchJobStatus |
|---------------|----------------|
| Submitted | pending |
| InProgress | running |
| Completed | succeeded |
| Failed | failed |
| Stopped | cancelled |
| Stopping | cancelled |
| PartiallyCompleted | partial |
| Scheduled | pending |
| Expired | failed |

**Input Format** (JSONL for Claude):

```json
{"recordId": "1", "modelInput": {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello"}]}}
```

**Input Format** (JSONL for Titan Embeddings):

```json
{"recordId": "1", "modelInput": {"inputText": "Text to embed"}}
```

**S3 Handling**:
- Bedrock uses S3 URIs directly (like VertexAI uses GCS)
- Need IAM role with S3 access
- User provides URIs (consistent with our approach)

### Implementation Notes

1. Requires `roleArn` - may need to add to options or driver config
2. Different input formats per model family (Claude, Titan, etc.)
3. Consider model-specific input formatters like VertexAI

---

## Implementation Checklist

### For Each Provider

- [ ] Create `batch/types.ts` with:
  - [ ] Provider-specific request/response types
  - [ ] Job state mapping function
  - [ ] Job ID encoding/decoding (if multi-model routing needed)

- [ ] Create `batch/index.ts` with `{Provider}BatchClient`:
  - [ ] `createBatchJob(options)` → `BatchJob`
  - [ ] `getBatchJob(jobId)` → `BatchJob`
  - [ ] `listBatchJobs(options?)` → `ListBatchJobsResult`
  - [ ] `cancelBatchJob(jobId)` → `BatchJob`
  - [ ] `deleteBatchJob(jobId)` → `void` (if supported)
  - [ ] `waitForCompletion(jobId, pollInterval?, maxWait?)` → `BatchJob`

- [ ] Update driver `index.ts`:
  - [ ] Add `private batchClient` field
  - [ ] Add `getBatchClient()` method

- [ ] Add tests in `drivers/test/batch-{provider}.test.ts`

### Source/Destination Updates

The common types use generic field names. Provider implementations should map:

| Common Field | VertexAI | OpenAI | Bedrock |
|--------------|----------|--------|---------|
| `source.gcsUris` | GCS URIs | N/A (file upload) | N/A |
| `source.s3Uris` | N/A | N/A | S3 URIs |
| `source.fileId` | N/A | File ID | N/A |
| `destination.gcsUri` | GCS URI | N/A | N/A |
| `destination.s3Uri` | N/A | N/A | S3 URI |

Consider extending `BatchJobSource` / `BatchJobDestination` in common types:

```typescript
export interface BatchJobSource {
    gcsUris?: string[];      // VertexAI
    s3Uris?: string[];       // Bedrock
    fileId?: string;         // OpenAI (after upload)
    bigqueryUri?: string;    // VertexAI
}
```

---

## Reference: VertexAI Implementation

### File: `vertexai/batch/types.ts`

Key patterns:
- State mapping function
- Job ID encoding with provider prefix
- Provider-specific API types

### File: `vertexai/batch/gemini-batch.ts`

Key patterns:
- SDK-based implementation
- Mapping SDK response to common `BatchJob`
- Handling completion stats type conversion

### File: `vertexai/batch/index.ts`

Key patterns:
- Model routing based on name patterns
- Unified interface across model types
- `waitForCompletion` polling implementation

---

## Testing

Environment variables needed:

```bash
# OpenAI
OPENAI_API_KEY=...

# Bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::...  # For batch jobs
BEDROCK_BATCH_S3_BUCKET=...              # For input/output
```

Test file structure:

```typescript
describe('Bedrock Batch', () => {
    it('should create inference batch job', async () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const batchClient = driver.getBatchClient();

        const job = await batchClient.createBatchJob({
            model: 'anthropic.claude-3-haiku-20240307-v1:0',
            type: BatchJobType.inference,
            source: { s3Uris: ['s3://bucket/input.jsonl'] },
            destination: { s3Uri: 's3://bucket/output/' },
        });

        expect(job.status).toBe(BatchJobStatus.pending);
    });
});
```
