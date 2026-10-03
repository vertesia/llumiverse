import Anthropic from '@anthropic-ai/sdk';
import {
    appendConversationRecords,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    fingerprintJson,
    type ModelTarget,
} from '@llumiverse/conversation';
import { resolveCanonicalExecutionContextOptions } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { countAnthropicModelSwitchRequest } from './canonical-model-switch-count.js';
import { AnthropicDriver } from './index.js';

const target: ModelTarget = {
    provider: 'anthropic',
    protocol: 'anthropic.messages',
    model: 'claude-sonnet-4-20250514',
    adapter_version: '2026-09-11.canonical.1',
};
const native = {
    model: target.model,
    messages: [{ role: 'user', content: [{ type: 'text', text: 'Retained text.' }] }],
    system: [{ type: 'text', text: 'Be concise.' }],
    max_tokens: 128,
    stream: false,
};

describe('configured Anthropic model-switch token count', () => {
    it('counts only the native text input through the non-generation endpoint', async () => {
        const client = new Anthropic({ apiKey: 'test-only' });
        const countTokens = vi.spyOn(client.messages, 'countTokens').mockResolvedValue({ input_tokens: 23 });
        const result = await countAnthropicModelSwitchRequest(client, native, target);
        expect(result).toEqual({
            status: 'counted',
            input_tokens: 23,
            profile: 'anthropic.messages.count_tokens:v1',
        });
        expect(countTokens).toHaveBeenCalledOnce();
        expect(countTokens.mock.calls[0]?.[0]).toEqual({
            model: target.model,
            messages: native.messages,
            system: native.system,
        });
    });

    it('leaves media, tools, altered model and unknown native fields unavailable without provider I/O', async () => {
        const client = new Anthropic({ apiKey: 'test-only' });
        const countTokens = vi.spyOn(client.messages, 'countTokens').mockResolvedValue({ input_tokens: 23 });
        const altered = [
            { ...native, model: 'other-model' },
            { ...native, tools: [{ name: 'fetch' }] },
            { ...native, messages: [{ role: 'user', content: [{ type: 'image', source: 'untrusted' }] }] },
            { ...native, previous_messages: ['hidden'] },
        ];
        for (const body of altered) {
            expect((await countAnthropicModelSwitchRequest(client, body, target)).status).toBe('unavailable');
        }
        expect(countTokens).not.toHaveBeenCalled();
    });

    it('rejects a failed count before Claude prepared publication or generation transport', async () => {
        const at = '2026-10-03T00:00:00.000Z';
        const empty = createConversationDocument({ id: 'conversation:provider-count', created_at: at });
        const turn = createUserTurn({
            id: 'turn:provider-count',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [createTextBlock({ id: 'block:provider-count', text: 'Retained text.', format: 'plain' })],
        });
        const source = appendConversationRecords(
            empty,
            { turns: [turn], context_entries: [{ id: 'entry:provider-count', type: 'source_turn', turn_id: turn.id }] },
            {
                expected_revision: empty.revision,
                operation_id: 'append:provider-count',
                payload_fingerprint: await fingerprintJson({ text: 'Retained text.' }),
                recorded_at: at,
            },
        ).document;
        const driver = new AnthropicDriver({ apiKey: 'test-only' });
        const transport = vi.spyOn(driver.client.messages, 'stream');
        const prepared = vi.fn(async () => undefined);
        const projected = vi.fn(async () => {
            throw new Error('Provider count unavailable');
        });
        const options = resolveCanonicalExecutionContextOptions({
            model: target.model,
            conversation: source,
            conversation_runtime: {
                conversation_id: source.id,
                request_id: 'request:provider-count',
                attempt_id: 'attempt:provider-count',
                input_operation_id: 'input:provider-count',
                response_operation_id: 'response:provider-count',
                recorded_at: at,
            },
            on_canonical_request_projected: projected,
            on_canonical_request_prepared: prepared,
        });
        await expect(driver.executeCanonicalContext(options)).rejects.toThrow('Provider count unavailable');
        expect(projected).toHaveBeenCalledOnce();
        expect(prepared).not.toHaveBeenCalled();
        expect(transport).not.toHaveBeenCalled();
        await expect(
            driver.streamCanonicalContextEvents(options, undefined, { stream_id: 'stream:provider-count' }),
        ).rejects.toThrow('Provider count unavailable');
        expect(projected).toHaveBeenCalledTimes(2);
        expect(prepared).not.toHaveBeenCalled();
        expect(transport).not.toHaveBeenCalled();
    });

    it('propagates cancellation before token-counter transport', async () => {
        const client = new Anthropic({ apiKey: 'test-only' });
        const countTokens = vi.spyOn(client.messages, 'countTokens').mockResolvedValue({ input_tokens: 23 });
        const controller = new AbortController();
        controller.abort(new Error('Cancelled plan'));
        await expect(countAnthropicModelSwitchRequest(client, native, target, controller.signal)).rejects.toThrow(
            'Cancelled plan',
        );
        expect(countTokens).not.toHaveBeenCalled();
    });
});
