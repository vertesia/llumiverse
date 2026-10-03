import type { ExecutionOptions } from '@llumiverse/common';
import { createConversationDocument } from '@llumiverse/conversation';
import { describe, expect, it, vi } from 'vitest';
import {
    type CanonicalExecutionContextInputOptions,
    resolveCanonicalExecutionContextOptions,
    resolveCanonicalExecutionOptions,
} from './CanonicalExecution.js';

describe('canonical runtime projection callback forwarding', () => {
    it('retains the same runtime callback through authoring and materialized option boundaries without invoking it', () => {
        const callback = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_projected']>>(async () => undefined);
        const options: CanonicalExecutionContextInputOptions = {
            model: 'gpt-4o-2024-08-06',
            conversation: createConversationDocument({ id: 'projection:core', created_at: '2026-10-02T00:00:00.000Z' }),
            conversation_runtime: {
                conversation_id: 'projection:core',
                request_id: 'request:core',
                attempt_id: 'attempt:core',
                input_operation_id: 'input:core',
                response_operation_id: 'response:core',
                recorded_at: '2026-10-02T00:00:00.000Z',
                started_at: '2026-10-02T00:00:00.000Z',
                purpose: 'interaction',
            },
            on_canonical_request_projected: callback,
        };
        expect(resolveCanonicalExecutionOptions(options).on_canonical_request_projected).toBe(callback);
        expect(resolveCanonicalExecutionContextOptions(options).on_canonical_request_projected).toBe(callback);
        expect(callback).not.toHaveBeenCalled();
    });
});
