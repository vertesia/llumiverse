import type { ExecutionOptions } from '@llumiverse/common';
import type { AgentContentBlock, DecodedConversationResponse } from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    assertDecodedCanonicalToolSelection,
    CanonicalToolSelectionViolationError,
    canonicalToolSelectionPolicy,
    parseCanonicalToolSelectionPolicy,
} from './CanonicalSelection.js';

function decoded(...toolNames: string[]): DecodedConversationResponse {
    const blocks: AgentContentBlock[] = toolNames.map((toolName, index) => ({
        id: `block-${index}`,
        type: 'tool_call',
        call_id: `call-${index}`,
        tool_name: toolName,
        executor: 'application',
        arguments: { type: 'json', value: {} },
    }));
    return {
        turns: [
            {
                id: 'turn',
                kind: 'agent',
                authority: 'ordinary',
                blocks,
                status: 'completed',
                timestamps: { recorded_at: '2026-10-01T00:00:00.000Z', completed_at: '2026-10-01T00:00:00.000Z' },
                provenance: { type: 'generated' },
                model_visibility: 'include',
                generation_id: 'generation',
            },
        ],
        generation: {
            id: 'generation',
            record_source: 'executed',
            request_id: 'request',
            attempt_id: 'attempt',
            purpose: 'interaction',
            requested_model: 'model',
            provider: 'provider',
            protocol: 'protocol',
            adapter_version: 'adapter',
            status: 'completed',
            timestamps: { recorded_at: '2026-10-01T00:00:00.000Z', completed_at: '2026-10-01T00:00:00.000Z' },
            source: { conversation_id: 'conversation', revision: 0 },
            request_receipt: {
                id: 'receipt',
                request_id: 'request',
                attempt_id: 'attempt',
                source: { conversation_id: 'conversation', revision: 0 },
                context_fingerprint: 'sha256:context',
                tool_set_fingerprint: 'sha256:tools',
                request_fingerprint: 'sha256:request',
                target: { provider: 'provider', protocol: 'protocol', model: 'model', adapter_version: 'adapter' },
                tool_definition_ids: [],
                asset_versions: [],
                item_mappings: [],
                recorded_at: '2026-10-01T00:00:00.000Z',
            },
        },
        diagnostics: [],
        payload_fingerprint: 'sha256:response',
    };
}

describe('canonical tool selection', () => {
    it('normalizes only explicit effective selection', () => {
        expect(canonicalToolSelectionPolicy({})).toBeUndefined();
        expect(canonicalToolSelectionPolicy({ model_options: { tool_choice: 'auto' } })).toEqual({ mode: 'auto' });
        expect(canonicalToolSelectionPolicy({ model_options: { tool_choice: 'none' } })).toEqual({ mode: 'none' });
        expect(canonicalToolSelectionPolicy({ model_options: { tool_choice: 'any' } })).toEqual({ mode: 'required' });
        expect(
            canonicalToolSelectionPolicy({
                model_options: {
                    tool_choice: 'none',
                    required_tool_name: 'required_tool',
                } as ExecutionOptions['model_options'] & { required_tool_name: string },
            }),
        ).toEqual({ mode: 'required', tool_name: 'required_tool' });
    });

    it('enforces none, unnamed required, and named required without excluding additional calls', () => {
        expect(() => assertDecodedCanonicalToolSelection(decoded(), { mode: 'none' })).not.toThrow();
        expect(() => assertDecodedCanonicalToolSelection(decoded('first'), { mode: 'none' })).toThrow(
            CanonicalToolSelectionViolationError,
        );
        expect(() => assertDecodedCanonicalToolSelection(decoded(), { mode: 'required' })).toThrow(
            CanonicalToolSelectionViolationError,
        );
        expect(() =>
            assertDecodedCanonicalToolSelection(decoded('other', 'required_tool'), {
                mode: 'required',
                tool_name: 'required_tool',
            }),
        ).not.toThrow();
        expect(() =>
            assertDecodedCanonicalToolSelection(decoded('other'), {
                mode: 'required',
                tool_name: 'required_tool',
            }),
        ).toThrow(CanonicalToolSelectionViolationError);
    });

    it('rejects a malformed present receipt marker', () => {
        expect(() => parseCanonicalToolSelectionPolicy({ mode: 'required', extra: true })).toThrow('unknown field');
        expect(() => parseCanonicalToolSelectionPolicy({ mode: 'none', tool_name: 'unexpected' })).toThrow(
            'invalid tool_name',
        );
        expect(() => parseCanonicalToolSelectionPolicy({ mode: 'required', tool_name: '' })).toThrow(
            'invalid tool_name',
        );
    });
});
