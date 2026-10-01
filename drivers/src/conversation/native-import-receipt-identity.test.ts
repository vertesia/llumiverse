import { appendConversationRecords, deriveConversationId, fingerprintJson } from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    BEDROCK_CONVERSE_PROTOCOL,
    bedrockConverseJsonValue,
    compileBedrockConverseConversation,
    exportLegacyBedrockConverseConversation,
    importBedrockConverseConversation,
    importBedrockConverseHistory,
} from '../bedrock/bedrock-converse-conversation-adapter.js';
import { importOpenAIChatCompletionsHistory } from './index.js';

const origin = {
    conversation_id: 'import-receipt-identity',
    recorded_at: '2026-10-01T00:00:00.000Z',
    provider: 'recorded-provider',
    model: 'recorded-model',
    completeness: 'complete' as const,
    tool_definitions: [
        {
            id: 'lookup:v1',
            name: 'lookup',
            version: '1',
            input_schema: { type: 'object', description: 'original-schema' },
        },
    ],
};
const history = {
    _is_openai_chat_completions: true,
    messages: [{ role: 'assistant', content: 'answer', reasoning_content: 'protected reasoning' }],
};

describe('native import receipt binds declared semantics', () => {
    it.each([
        { name: 'provider', options: { ...origin, provider: 'another-provider' } },
        { name: 'model', options: { ...origin, model: 'another-model' } },
        {
            name: 'tool schema',
            options: {
                ...origin,
                tool_definitions: [
                    {
                        ...origin.tool_definitions[0],
                        input_schema: { type: 'object', description: 'different-schema' },
                    },
                ],
            },
        },
        { name: 'completeness', options: { ...origin, completeness: 'fragment' as const } },
        { name: 'timestamp', options: { ...origin, recorded_at: '2026-10-02T00:00:00.000Z' } },
        { name: 'source identity', options: { ...origin, source_request_id: 'recorded-other-request' } },
    ])('rejects replay of a conflicting declared $name under the same operation identity', async ({ options }) => {
        const first = await importOpenAIChatCompletionsHistory(history, origin);
        const second = await importOpenAIChatCompletionsHistory(history, options);
        const firstReceipt = Object.values(first.document.operation_receipts)[0];
        const secondReceipt = Object.values(second.document.operation_receipts)[0];
        expect(secondReceipt.id).toBe(firstReceipt.id);
        expect(secondReceipt.payload_fingerprint).not.toBe(firstReceipt.payload_fingerprint);
        expect(() =>
            appendConversationRecords(
                first.document,
                {
                    turns: second.document.turns,
                    tool_definitions: Object.values(second.document.tool_definitions),
                    context_entries: second.document.context.entries,
                },
                {
                    expected_revision: first.document.revision,
                    operation_id: secondReceipt.id,
                    payload_fingerprint: secondReceipt.payload_fingerprint,
                    recorded_at: origin.recorded_at,
                },
            ),
        ).toThrow(/already used with a different payload/);
        if (options.completeness === origin.completeness) expect(second.document).not.toEqual(first.document);
        else expect(second.report.completeness).not.toEqual(first.report.completeness);
    });

    it('returns the same receipt for an exact options and history retry', async () => {
        const first = await importOpenAIChatCompletionsHistory(history, origin);
        const second = await importOpenAIChatCompletionsHistory(structuredClone(history), structuredClone(origin));
        expect(second).toEqual(first);
        const receipt = Object.values(second.document.operation_receipts)[0];
        const retry = appendConversationRecords(
            first.document,
            {
                turns: second.document.turns,
                tool_definitions: Object.values(second.document.tool_definitions),
                context_entries: second.document.context.entries,
            },
            {
                expected_revision: first.document.revision,
                operation_id: receipt.id,
                payload_fingerprint: receipt.payload_fingerprint,
                recorded_at: origin.recorded_at,
            },
        );
        expect(retry.applied).toBe(false);
        expect(retry.document).toEqual(first.document);
    });
});

it('normalizes omitted defaults that do not change document, receipt or report', async () => {
    const { completeness: _completeness, tool_definitions: _tools, ...minimal } = origin;
    const first = await importOpenAIChatCompletionsHistory(history, minimal);
    const second = await importOpenAIChatCompletionsHistory(history, {
        ...minimal,
        completeness: 'unknown',
        source_request_id: `${minimal.conversation_id}:legacy`,
        tool_definitions: [],
    });
    expect(second).toEqual(first);
});

it('keeps the named historical Bedrock document-only import receipt and replay boundary explicit', async () => {
    const native = {
        messages: [
            {
                role: 'assistant',
                content: [
                    {
                        reasoningContent: {
                            reasoningText: { text: 'recorded reasoning', signature: 'recorded signature' },
                        },
                    },
                    { text: 'answer' },
                ],
            },
        ],
    };
    const options = { ...origin, provider: 'bedrock', model: 'anthropic.claude-sonnet-4-6' };
    const legacy = await importBedrockConverseConversation(native, options);
    const receipt = Object.values(legacy.operation_receipts)[0];
    expect(receipt.id).toBe(
        await deriveConversationId('operation', options.conversation_id, BEDROCK_CONVERSE_PROTOCOL, 'import'),
    );
    expect(receipt.payload_fingerprint).toBe(await fingerprintJson(bedrockConverseJsonValue(native)));
    expect(exportLegacyBedrockConverseConversation(legacy)).toEqual(native);
    expect(
        compileBedrockConverseConversation(legacy, { provider: options.provider, model: options.model }).conversation,
    ).toEqual(native);
    const current = await importBedrockConverseHistory(native, options);
    expect(Object.values(current.document.operation_receipts)[0].payload_fingerprint).not.toBe(
        receipt.payload_fingerprint,
    );
});
