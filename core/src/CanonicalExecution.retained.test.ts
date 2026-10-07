import { type ConversationPreparedRequestRecord, createConversationDocument } from '@llumiverse/conversation';
import { describe, expect, expectTypeOf, it } from 'vitest';
import {
    CANONICAL_RETAINED_PREPARED_REQUEST,
    type CanonicalExecutionContextInputOptions,
    canonicalRetainedPreparedRequest,
    resolveCanonicalExecutionContextOptions,
    resolveCanonicalExecutionOptions,
    withCanonicalRetainedPreparedRequest,
} from './CanonicalExecution.js';
import type { Driver } from './Driver.js';

const at = '2026-10-03T00:00:00.000Z';
function fixture() {
    const runtime = {
        conversation_id: 'conversation',
        request_id: 'request',
        attempt_id: 'attempt',
        input_operation_id: 'input',
        response_operation_id: 'response',
        recorded_at: at,
        purpose: 'conversation',
        materialized_input: { operation_id: 'input', result_revision: 1 },
    };
    const record: ConversationPreparedRequestRecord = {
        source: { conversation_id: 'conversation', revision: 2 },
        runtime,
        generation_id: 'generation',
        response_turn_id: 'turn',
        request_receipt: {
            id: 'request-receipt',
            request_id: 'request',
            attempt_id: 'attempt',
            source: { conversation_id: 'conversation', revision: 2 },
            context_fingerprint: 'context',
            tool_set_fingerprint: 'tools',
            request_fingerprint: 'request',
            target: { provider: 'test', protocol: 'test', model: 'model', adapter_version: 'v1' },
            tool_definition_ids: [],
            asset_versions: [],
            item_mappings: [],
            recorded_at: at,
        },
    };
    const options: CanonicalExecutionContextInputOptions = {
        model: 'model',
        conversation: createConversationDocument({ id: 'conversation', created_at: at }),
        conversation_runtime: runtime,
    };
    return { options, record };
}

describe('native context retained prepared-record boundary', () => {
    it('copies evidence only across the explicit native boundary, excluding JSON and ordinary spreads', () => {
        const { options, record } = fixture();
        const attached = withCanonicalRetainedPreparedRequest(options, record);
        const normalized = resolveCanonicalExecutionContextOptions(attached);
        expect(canonicalRetainedPreparedRequest(normalized)).toEqual(record);
        expect(Object.getOwnPropertyDescriptor(normalized, CANONICAL_RETAINED_PREPARED_REQUEST)?.enumerable).toBe(
            false,
        );
        expect(canonicalRetainedPreparedRequest({ ...normalized })).toBeUndefined();
        expect(JSON.stringify(attached)).toBe(JSON.stringify(options));
        expect(Reflect.ownKeys(options)).not.toContain(CANONICAL_RETAINED_PREPARED_REQUEST);
        record.runtime.request_id = 'changed';
        options.conversation_runtime.request_id = 'changed';
        expect(canonicalRetainedPreparedRequest(normalized)?.runtime.request_id).toBe('request');
        expect(normalized.conversation_runtime.request_id).toBe('request');
    });

    it('owns optional virtual-parent options while retaining concrete provider model contracts', () => {
        const { options, record } = fixture();
        const virtualOptions: CanonicalExecutionContextInputOptions<string | undefined> = {
            ...options,
            model: undefined,
        };
        const attached = withCanonicalRetainedPreparedRequest(virtualOptions, record);
        const normalized = resolveCanonicalExecutionContextOptions(attached);
        expect(normalized.model).toBeUndefined();
        expect(canonicalRetainedPreparedRequest(normalized)).toEqual(record);
        record.runtime.request_id = 'changed';
        virtualOptions.conversation_runtime.request_id = 'changed';
        expect(normalized.conversation_runtime.request_id).toBe('request');
        expect(canonicalRetainedPreparedRequest(normalized)?.runtime.request_id).toBe('request');
        expect(Object.getOwnPropertyDescriptor(normalized, CANONICAL_RETAINED_PREPARED_REQUEST)?.enumerable).toBe(
            false,
        );
        expect(JSON.stringify(normalized)).not.toContain('request_receipt');
        expectTypeOf(normalized.model).toEqualTypeOf<string | undefined>();
        expectTypeOf<CanonicalExecutionContextInputOptions['model']>().toEqualTypeOf<string>();
        expectTypeOf<Parameters<Driver['executeCanonicalContext']>[0]['model']>().toEqualTypeOf<string>();
        expectTypeOf<Parameters<Driver['streamCanonicalContextEvents']>[0]['model']>().toEqualTypeOf<string>();
        const concrete = resolveCanonicalExecutionContextOptions<string>({ ...options, model: 'concrete-child' });
        expect(concrete.model).toBe('concrete-child');
        expectTypeOf(concrete.model).toEqualTypeOf<string>();
    });

    it('does not promote a caller-carried retained record through ordinary canonical authoring', () => {
        const { options, record } = fixture();
        const supplied = withCanonicalRetainedPreparedRequest(options, record);
        const ordinary = resolveCanonicalExecutionOptions(supplied);
        expect(canonicalRetainedPreparedRequest(ordinary)).toBeUndefined();
        expect(ordinary.conversation_runtime.request_id).toBe(options.conversation_runtime.request_id);
    });

    it('rejects accessor evidence without invoking it', () => {
        const { options } = fixture();
        let accessed = false;
        Object.defineProperty(options, CANONICAL_RETAINED_PREPARED_REQUEST, {
            get() {
                accessed = true;
                throw new Error('must not run');
            },
        });
        expect(() => resolveCanonicalExecutionContextOptions(options)).toThrow(/own data property/);
        expect(accessed).toBe(false);
    });

    it('uses the strict authoritative record parser and existing finite JSON bounds', () => {
        const { options, record } = fixture();
        const extra = { ...record, injected: true };
        expect(() => withCanonicalRetainedPreparedRequest(options, extra)).toThrow();
        const cyclic = { ...record };
        Object.defineProperty(cyclic, 'extra', { value: cyclic, enumerable: true });
        expect(() => withCanonicalRetainedPreparedRequest(options, cyclic)).toThrow(/preflight/);
        const huge = { ...record, response_turn_id: 'x'.repeat(32 * 1024 * 1024 + 1) };
        expect(() => withCanonicalRetainedPreparedRequest(options, huge)).toThrow(/preflight/);
    });
});
