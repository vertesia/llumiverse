import { fingerprintJson, type JsonValue } from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    importBedrockConverseHistory,
    importClaudeMessagesHistory,
    importGeminiGenerateContentHistory,
    importOpenAIChatCompletionsHistory,
    importOpenAIResponsesHistory,
} from './index.js';

function origin() {
    return {
        conversation_id: 'snapshot-source',
        recorded_at: '2026-10-01T00:00:00.000Z',
        provider: 'recorded-host',
        model: 'recorded-model',
        completeness: 'complete' as 'complete' | 'fragment',
        tool_definitions: [
            {
                id: 'lookup:v1',
                name: 'lookup',
                version: '1',
                input_schema: { type: 'object', description: 'recorded-schema' },
            },
        ],
    };
}
const cases = [
    {
        name: 'bedrock',
        importer: importBedrockConverseHistory,
        fixture: () => {
            const history = {
                messages: [
                    { role: 'user', content: [{ text: 'recorded-first' }] },
                    { role: 'assistant', content: [{ text: 'recorded-last' }] },
                ],
            };
            return {
                history,
                mutate: (message = 'caller-mutated') => {
                    history.messages[1].content[0].text = message;
                },
            };
        },
    },

    {
        name: 'chat',
        importer: importOpenAIChatCompletionsHistory,
        fixture: () => {
            const history = {
                _is_openai_chat_completions: true,
                messages: [
                    { role: 'user', content: 'recorded-first' },
                    { role: 'assistant', content: 'recorded-last' },
                ],
            };
            return {
                history,
                mutate: (message = 'caller-mutated') => {
                    history.messages[1].content = message;
                },
            };
        },
    },
    {
        name: 'responses',
        importer: importOpenAIResponsesHistory,
        fixture: () => {
            const history = [
                { role: 'user', content: 'recorded-first' },
                { role: 'assistant', content: 'recorded-last' },
            ];
            return {
                history,
                mutate: (message = 'caller-mutated') => {
                    history[1].content = message;
                },
            };
        },
    },
    {
        name: 'claude',
        importer: importClaudeMessagesHistory,
        fixture: () => {
            const history = {
                messages: [
                    { role: 'user', content: 'recorded-first' },
                    { role: 'assistant', content: 'recorded-last' },
                ],
            };
            return {
                history,
                mutate: (message = 'caller-mutated') => {
                    history.messages[1].content = message;
                },
            };
        },
    },
    {
        name: 'gemini',
        importer: importGeminiGenerateContentHistory,
        fixture: () => {
            const history = [
                { role: 'user', parts: [{ text: 'recorded-first' }] },
                { role: 'model', parts: [{ text: 'recorded-last' }] },
            ];
            return {
                history,
                mutate: (message = 'caller-mutated') => {
                    history[1].parts[0].text = message;
                },
            };
        },
    },
];
describe('native import owns a snapshot before the first await', () => {
    it.each(cases)(
        'isolates $name caller changes and fingerprints the exact recorded history',
        async ({ importer, fixture }) => {
            const { history, mutate } = fixture();
            const before = structuredClone(history);
            const options = origin();
            const beforeOptions = structuredClone(options);
            const pending = importer(history, options);
            mutate();
            options.provider = 'caller-host';
            options.model = 'caller-model';
            options.completeness = 'fragment';
            options.tool_definitions[0].input_schema.description = 'caller-schema';
            await Promise.resolve();
            mutate('after-await-mutation');
            options.provider = 'after-await-provider';
            options.model = 'after-await-model';
            options.tool_definitions[0].input_schema.description = 'after-await-schema';
            const result = await pending;
            expect(JSON.stringify(result.document)).toContain('recorded-last');
            expect(JSON.stringify(result.document)).not.toContain('caller-mutated');
            expect(JSON.stringify(result.document)).not.toContain('after-await');
            expect(result.report.completeness).toBe('complete');
            expect(result.document.tool_definitions['lookup:v1'].input_schema).toMatchObject({
                description: 'recorded-schema',
            });
            expect(Object.values(result.document.operation_receipts)[0]?.payload_fingerprint).toBe(
                await fingerprintJson({
                    format: 'llumiverse.native-conversation-import/v1',
                    protocol: result.report.protocol,
                    adapter_version: result.report.adapter_version,
                    history: before as JsonValue,
                    options: { ...beforeOptions, source_request_id: `${beforeOptions.conversation_id}:legacy` },
                }),
            );
        },
    );
});

describe('Bedrock owned native snapshot', () => {
    it('pins source provider/model and copies binary content before caller mutation', async () => {
        const bytes = Buffer.from([97, 98, 99]);
        const history = {
            messages: [
                { role: 'user', content: [{ image: { format: 'png', source: { bytes } } }] },
                {
                    role: 'assistant',
                    content: [
                        {
                            reasoningContent: {
                                reasoningText: { text: 'recorded-thinking', signature: 'recorded-signature' },
                            },
                        },
                    ],
                },
            ],
        };
        const options = origin();
        const pending = importBedrockConverseHistory(history, options);
        bytes.fill(0);
        options.provider = 'caller-host';
        options.model = 'caller-model';
        const result = await pending;
        expect(Object.values(result.document.assets)[0]).toMatchObject({
            storage: { type: 'inline_base64', data: 'YWJj' },
        });
        expect(JSON.stringify(result.document)).toContain('recorded-thinking');
        expect(JSON.stringify(result.document)).not.toContain('caller-host');
        expect(JSON.stringify(result.document)).not.toContain('caller-model');
        const replay = result.document.turns[1].blocks.find((block) => block.type === 'native_replay');
        expect(replay).toMatchObject({ compatibility_scope: { provider: 'recorded-host', model: 'recorded-model' } });
    });
});
