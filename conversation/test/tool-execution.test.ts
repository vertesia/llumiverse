import { describe, expect, it, vi } from 'vitest';
import {
    appendConversationRecords,
    appendToolExecutionResult,
    createConversationDocument,
    fingerprintJson,
    resolveToolExecutionRequest,
    setProcessingPolicy,
    type ToolCallSourceRef,
    validateToolExecutionResult,
} from '../src/index.js';

const RECORDED_AT = '2026-09-30T00:00:00Z';

function callDocument(executor: 'application' | 'provider' = 'application') {
    const document = createConversationDocument({ id: 'conversation', created_at: RECORDED_AT });
    document.turns.push({
        id: 'agent-turn',
        kind: 'agent',
        authority: 'ordinary',
        blocks: [
            {
                id: 'call-block',
                type: 'tool_call',
                call_id: 'call-1',
                tool_name: 'write_artifact',
                executor,
                arguments: { type: 'json', value: { name: 'report.md', content: 'exact body' } },
            },
        ],
        status: 'completed',
        timestamps: { recorded_at: RECORDED_AT },
        provenance: { type: 'imported', source: 'test' },
        model_visibility: 'include',
    });
    document.context.entries.push({ id: 'agent-context', type: 'source_turn', turn_id: 'agent-turn' });
    return document;
}

async function callSource(
    document: ReturnType<typeof callDocument>,
    revision = document.revision,
): Promise<ToolCallSourceRef> {
    const call = document.turns[0].blocks[0];
    return {
        conversation: { conversation_id: 'conversation', revision },
        turn_id: 'agent-turn',
        block_id: 'call-block',
        call_id: 'call-1',
        call_fingerprint: await fingerprintJson(call),
    };
}

describe('canonical tool execution', () => {
    it('resolves exact JSON for an application call without persisting another argument copy', async () => {
        const document = callDocument();
        const resolver = vi.fn();
        const source = await callSource(document);

        await expect(resolveToolExecutionRequest(document, source, resolver)).resolves.toEqual({
            source,
            call: document.turns[0].blocks[0],
            arguments: { name: 'report.md', content: 'exact body' },
        });
        expect(resolver).not.toHaveBeenCalled();
        expect(document.turns[0].blocks[0]).toHaveProperty('arguments.type', 'json');
    });

    it('rejects provider-owned calls and a stale mutable document head', async () => {
        const resolver = vi.fn();
        const providerDocument = callDocument('provider');
        await expect(
            resolveToolExecutionRequest(providerDocument, await callSource(providerDocument), resolver),
        ).rejects.toThrow('not authorized for application execution');
        const document = callDocument();
        const advanced = { ...document, revision: 1, context: { ...document.context, revision: 1 } };
        await expect(resolveToolExecutionRequest(advanced, await callSource(document), resolver)).rejects.toThrow(
            'requires conversation revision 0, received 1',
        );
    });

    it('refuses to execute a call that already has a terminal receipt', async () => {
        const document = callDocument();
        const source = await callSource(document);
        document.execution_receipts.done = {
            id: 'done',
            call_id: source.call_id,
            executor: 'application',
            status: 'denied',
            result_fingerprint: 'sha256:denied',
            recorded_at: RECORDED_AT,
            call_source: source,
        };

        await expect(resolveToolExecutionRequest(document, source, vi.fn())).rejects.toThrow(
            'already has a terminal result',
        );
    });

    it('appends a full tool turn, result asset, and execution receipt after an independent revision advance', async () => {
        const initial = callDocument();
        const advanced = appendConversationRecords(
            initial,
            {
                turns: [
                    {
                        id: 'parallel-user-turn',
                        kind: 'user',
                        authority: 'ordinary',
                        blocks: [{ id: 'parallel-text', type: 'text', text: 'parallel input', format: 'plain' }],
                        status: 'completed',
                        timestamps: { recorded_at: RECORDED_AT },
                        provenance: { type: 'received' },
                        model_visibility: 'include',
                    },
                ],
                context_entries: [{ id: 'parallel-user-context', type: 'source_turn', turn_id: 'parallel-user-turn' }],
            },
            {
                expected_revision: 0,
                operation_id: 'parallel-input',
                payload_fingerprint: 'sha256:parallel-input',
                recorded_at: RECORDED_AT,
            },
        ).document;
        const resultBlock = {
            id: 'result-block',
            type: 'tool_result' as const,
            call_id: 'call-1',
            status: 'success' as const,
            content: [{ id: 'result-image', type: 'image' as const, asset_id: 'result-asset' }],
        };
        const source = await callSource(initial, 0);
        const result = {
            source,
            turn: {
                id: 'tool-turn',
                kind: 'tool' as const,
                authority: 'ordinary' as const,
                blocks: [resultBlock] as [typeof resultBlock],
                status: 'completed' as const,
                timestamps: { recorded_at: RECORDED_AT },
                execution_id: 'execution-1',
                provenance: { type: 'received' as const },
                model_visibility: 'include' as const,
            },
            assets: [
                {
                    id: 'result-asset',
                    kind: 'image' as const,
                    mime_type: 'image/png',
                    storage: { type: 'inline_base64' as const, data: 'AA==' },
                    provenance: { type: 'received' as const, source_turn_id: 'tool-turn' },
                    byte_length: 1,
                    created_at: RECORDED_AT,
                },
            ],
            execution_receipt: {
                id: 'execution-1',
                call_id: 'call-1',
                executor: 'application' as const,
                status: 'success' as const,
                result_turn_id: 'tool-turn',
                result_fingerprint: await fingerprintJson(resultBlock),
                recorded_at: RECORDED_AT,
                call_source: source,
            },
        };

        const beforeValidation = structuredClone(advanced);
        await expect(validateToolExecutionResult(advanced, result)).resolves.toEqual(result);
        expect(advanced).toEqual(beforeValidation);

        const accepted = await appendToolExecutionResult(advanced, result, {
            expected_revision: 1,
            operation_id: 'tool-result-1',
            recorded_at: RECORDED_AT,
        });

        expect(accepted.applied).toBe(true);
        expect(accepted.document.revision).toBe(2);
        expect(accepted.document.turns.at(-1)).toEqual(result.turn);
        expect(accepted.document.assets['result-asset']).toEqual(result.assets[0]);
        expect(accepted.document.execution_receipts['execution-1']).toEqual(result.execution_receipt);

        const retried = await appendToolExecutionResult(accepted.document, result, {
            expected_revision: 1,
            operation_id: 'tool-result-1',
            recorded_at: RECORDED_AT,
        });
        expect(retried.applied).toBe(false);
        expect(retried.document.revision).toBe(2);
    });

    it('rejects result evidence that does not match its retained call or result block', async () => {
        const document = callDocument();
        const resultBlock = {
            id: 'result-block',
            type: 'tool_result' as const,
            call_id: 'call-1',
            status: 'success' as const,
            content: [{ id: 'result-text', type: 'text' as const, text: 'done', format: 'plain' as const }],
        };
        const source = await callSource(document);
        const result = {
            source,
            turn: {
                id: 'tool-turn',
                kind: 'tool' as const,
                authority: 'ordinary' as const,
                blocks: [resultBlock] as [typeof resultBlock],
                status: 'completed' as const,
                timestamps: { recorded_at: RECORDED_AT },
                execution_id: 'execution-1',
                provenance: { type: 'received' as const },
                model_visibility: 'include' as const,
            },
            execution_receipt: {
                id: 'execution-1',
                call_id: 'call-1',
                executor: 'application' as const,
                status: 'success' as const,
                result_turn_id: 'tool-turn',
                result_fingerprint: 'sha256:not-the-result',
                recorded_at: RECORDED_AT,
                call_source: source,
            },
        };

        await expect(
            appendToolExecutionResult(document, result, {
                expected_revision: 0,
                operation_id: 'tool-result-1',
                recorded_at: RECORDED_AT,
            }),
        ).rejects.toThrow('result fingerprint does not match');

        // A receipt must attest to the very same call that the top-level source authorized.
        // A valid result hash must not let a different call fingerprint enter the execution ledger.
        const mismatchedSource = {
            ...result,
            execution_receipt: {
                ...result.execution_receipt,
                result_fingerprint: await fingerprintJson(resultBlock),
                call_source: { ...source, call_fingerprint: await fingerprintJson({ other: 'call' }) },
            },
        };
        await expect(
            appendToolExecutionResult(document, mismatchedSource, {
                expected_revision: 0,
                operation_id: 'tool-result-1',
                recorded_at: RECORDED_AT,
            }),
        ).rejects.toThrow('does not match application call');
        await expect(validateToolExecutionResult(document, mismatchedSource)).rejects.toThrow(
            'does not match application call',
        );
        await expect(validateToolExecutionResult(document, result)).rejects.toThrow(
            'result fingerprint does not match',
        );
        expect(document.execution_receipts).toEqual({});
    });
});

describe('application tool result processing policy', () => {
    it('keeps the async API and appends result, receipt and jobs atomically without changing the call source', async () => {
        const original = callDocument();
        const source = await callSource(original);
        const enabled = await setProcessingPolicy(original, {
            operation_id: 'enable',
            expected_revision: original.revision,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [
                {
                    id: 'externalize-text',
                    version: '1',
                    scope: 'on_append',
                    config: {},
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        });
        const block = {
            id: 'tool-result',
            type: 'tool_result' as const,
            call_id: source.call_id,
            status: 'success' as const,
            content: [
                { id: 'tool-text', type: 'text' as const, text: 'retained tool result', format: 'plain' as const },
            ],
        };
        const result = {
            source,
            turn: {
                id: 'tool-turn',
                kind: 'tool' as const,
                authority: 'ordinary' as const,
                model_visibility: 'include' as const,
                status: 'completed' as const,
                timestamps: { recorded_at: RECORDED_AT },
                provenance: { type: 'received' as const },
                execution_id: 'tool-execution',
                blocks: [block],
            },
            execution_receipt: {
                id: 'tool-execution',
                call_id: source.call_id,
                executor: 'application' as const,
                status: 'success' as const,
                result_turn_id: 'tool-turn',
                result_fingerprint: await fingerprintJson(block),
                recorded_at: RECORDED_AT,
                call_source: source,
            },
        };
        const options = {
            expected_revision: enabled.document.revision,
            operation_id: 'tool-append',
            recorded_at: RECORDED_AT,
        };
        const accepted = await appendToolExecutionResult(enabled.document, result, options);
        const receipt = accepted.document.operation_receipts['tool-append'];
        expect(receipt.accepted_execution_receipt_ids).toEqual(['tool-execution']);
        expect(accepted.document.execution_receipts['tool-execution']).toEqual(result.execution_receipt);
        expect(Object.keys(accepted.document.processing.jobs ?? {})).toHaveLength(1);
        const retry = await appendToolExecutionResult(accepted.document, result, options);
        expect(retry.applied).toBe(false);
        expect(retry.document).toEqual(accepted.document);
    });
});
