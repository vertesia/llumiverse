import { readFile } from 'node:fs/promises';
import { createConversationDocument, parseConversationDocument, setProcessingPolicy } from '@llumiverse/conversation';
import { isCanonicalAcceptedRecovery, PromptRole, Providers } from '@llumiverse/core';
import type OpenAI from 'openai';
import { expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from './index.js';

class ImageUpgradeDriver extends OpenAIResponsesDriverBase {
    provider: Providers.openai = Providers.openai;
    service: OpenAI;

    constructor(generate: (request: unknown, options?: { signal?: AbortSignal }) => Promise<unknown>) {
        super({});
        this.service = { images: { generate } } as unknown as OpenAI;
    }
}

it('holds an enabled direct image request before transport until its input is host processed', async () => {
    const at = '2026-09-30T03:00:00.000Z';
    const initial = createConversationDocument({ id: 'conversation:image-processing', created_at: at });
    const enabled = await setProcessingPolicy(initial, {
        operation_id: 'policy:image-processing',
        expected_revision: initial.revision,
        recorded_at: at,
        enabled: true,
        processors: [
            {
                id: 'noop',
                version: 'v1',
                scope: 'on_append',
                config: {},
                required: true,
                failure_behavior: 'block',
            },
        ],
    });
    const generate = vi.fn(async () => ({
        created: 1,
        data: [{ b64_json: 'YQ==' }],
        output_format: 'png',
    }));
    const driver = new ImageUpgradeDriver(generate);
    await expect(
        driver.executeCanonical([{ role: PromptRole.user, content: 'Draw an icon.' }], {
            model: 'gpt-image-1',
            model_options: { _option_id: 'openai-gpt-image', width: 1024, height: 1024, output_format: 'png' },
            conversation: enabled.document,
            conversation_runtime: {
                conversation_id: initial.id,
                request_id: 'request:image-processing',
                attempt_id: 'attempt:image-processing',
                input_operation_id: 'input:image-processing',
                response_operation_id: 'response:image-processing',
                recorded_at: at,
            },
        }),
    ).rejects.toThrow('requires appendConversationRecordsWithProcessing');
    expect(generate).not.toHaveBeenCalled();
});

it('recovers the exact serialized image artifact emitted by published provider44 without transport or recharge', async () => {
    const fixture = JSON.parse(
        await readFile(new URL('./fixtures/openai-image-published-v2.json', import.meta.url), 'utf8'),
    ) as {
        provenance: {
            source_commit: string;
            source_sha256: string;
            built_lib_sha256: string;
            openai_version: string;
        };
        conversation: unknown;
        accepted_output: { generation: { usage?: unknown } };
    };
    expect(fixture.provenance).toEqual({
        source_commit: '44a77a0119dc87d4f8fc757ec5b0ac3dae7cda41',
        source_sha256: '9f1ac5284e5ee7ad87d1e98389bf2194f6ae6f9c0f23ea6b70b2fe186a04fabb',
        built_lib_sha256: '28831dfffd78560f06c0cbe6287fbb53dc18f7e2df02e5c79d7c80f75160430e',
        openai_version: '7.23.0',
    });
    const conversation = parseConversationDocument(fixture.conversation);
    const acceptedGenerationId = Object.values(conversation.generations).find(
        (generation) => generation.adapter_version === '2026-09-30.canonical.2',
    )?.id;
    if (acceptedGenerationId === undefined) throw new Error('Published v2 fixture has no accepted image generation');
    const acceptedGeneration = conversation.generations[acceptedGenerationId];
    expect(acceptedGeneration?.request_receipt?.target.adapter_version).toBe('2026-09-30.canonical.2');

    const generate = vi.fn(async () => {
        throw new Error('Published v2 recovery must not open image transport');
    });
    const publish = vi.fn(async () => undefined);
    const driver = new ImageUpgradeDriver(generate);
    const result = await driver.executeCanonical([{ role: PromptRole.user, content: 'Draw a published v2 icon.' }], {
        model: 'gpt-image-1',
        model_options: {
            _option_id: 'openai-gpt-image',
            width: 1024,
            height: 1024,
            image_quality: 'high',
            output_format: 'png',
            n: 1,
        },
        conversation,
        conversation_runtime: {
            conversation_id: 'conversation:published-v2-upgrade',
            request_id: 'request:published-v2-upgrade',
            attempt_id: 'attempt:published-v2-upgrade:current-runtime',
            input_operation_id: 'input:published-v2-upgrade',
            response_operation_id: 'response:published-v2-upgrade',
            recorded_at: '2026-09-30T03:00:00.000Z',
        },
        on_canonical_request_prepared: publish,
    });

    expect(isCanonicalAcceptedRecovery(result)).toBe(true);
    expect(result.accepted_output.generation.adapter_version).toBe('2026-09-30.canonical.2');
    expect(result.accepted_output.generation.usage).toEqual(fixture.accepted_output.generation.usage);
    expect(result.conversation).toEqual(conversation);
    expect(generate).not.toHaveBeenCalled();
    expect(publish).not.toHaveBeenCalled();
});

it('context-recovers a published flattened system prompt but rejects it for a new image request', async () => {
    const fixture = JSON.parse(
        await readFile(new URL('./fixtures/openai-image-published-v2-system.json', import.meta.url), 'utf8'),
    ) as {
        provenance: {
            source_commit: string;
            source_sha256: string;
            built_lib_sha256: string;
            openai_version: string;
        };
        capturedNativePayload: unknown;
        conversation: unknown;
        accepted_output: { generation: { usage?: unknown } };
    };
    expect(fixture.provenance).toEqual({
        source_commit: '44a77a0119dc87d4f8fc757ec5b0ac3dae7cda41',
        source_sha256: '9f1ac5284e5ee7ad87d1e98389bf2194f6ae6f9c0f23ea6b70b2fe186a04fabb',
        built_lib_sha256: '28831dfffd78560f06c0cbe6287fbb53dc18f7e2df02e5c79d7c80f75160430e',
        openai_version: '7.23.0',
    });
    expect(fixture.capturedNativePayload).toEqual({
        model: 'gpt-image-1',
        prompt: 'Create a compact icon.\nDraw a published v2 icon.',
        size: '1024x1024',
        n: 1,
        quality: 'high',
        output_format: 'png',
    });
    const conversation = parseConversationDocument(fixture.conversation);
    expect(conversation.turns.slice(0, 2)).toEqual([
        expect.objectContaining({ kind: 'program', authority: 'system' }),
        expect.objectContaining({ kind: 'user', authority: 'ordinary' }),
    ]);

    const generate = vi.fn(async () => {
        throw new Error('Published system-prompt recovery must not open image transport');
    });
    const publish = vi.fn(async () => undefined);
    const driver = new ImageUpgradeDriver(generate);
    const options = {
        model: 'gpt-image-1',
        model_options: {
            _option_id: 'openai-gpt-image' as const,
            width: 1024,
            height: 1024,
            image_quality: 'high' as const,
            output_format: 'png' as const,
            n: 1,
        },
        conversation,
        conversation_runtime: {
            conversation_id: 'conversation:published-v2-upgrade',
            request_id: 'request:published-v2-upgrade',
            attempt_id: 'attempt:published-v2-upgrade:context-recovery',
            input_operation_id: 'input:published-v2-upgrade',
            response_operation_id: 'response:published-v2-upgrade',
            recorded_at: '2026-09-30T03:00:00.000Z',
        },
        on_canonical_request_prepared: publish,
    };

    const recovered = await driver.executeCanonicalContext(options);
    expect(isCanonicalAcceptedRecovery(recovered)).toBe(true);
    expect(recovered.accepted_output.generation.adapter_version).toBe('2026-09-30.canonical.2');
    expect(recovered.accepted_output.generation.usage).toEqual(fixture.accepted_output.generation.usage);
    expect(recovered.conversation).toEqual(conversation);
    expect(generate).not.toHaveBeenCalled();
    expect(publish).not.toHaveBeenCalled();

    await expect(
        driver.executeCanonicalContext({
            ...options,
            conversation_runtime: {
                ...options.conversation_runtime,
                request_id: 'request:published-v2-upgrade:new-context',
                attempt_id: 'attempt:published-v2-upgrade:new-context',
                input_operation_id: 'input:published-v2-upgrade:new-context',
                response_operation_id: 'response:published-v2-upgrade:new-context',
            },
        }),
    ).rejects.toThrow(/cannot preserve program turn .* authority system/);
    expect(generate).not.toHaveBeenCalled();
    expect(publish).not.toHaveBeenCalled();
});
