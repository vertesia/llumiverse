import type { ConverseRequest, ConverseResponse, Message } from '@aws-sdk/client-bedrock-runtime';
import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from './index.js';

const MODEL = 'arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.openai.gpt-6-astra';
const tools = ['preview_artifact_officexml', 'other_preview'].map((name) => ({
    name,
    input_schema: { type: 'object', properties: {} },
}));
const image = { format: 'jpeg' as const, source: { bytes: new Uint8Array([1, 2, 3]) } };

function messages(): Message[] {
    return [
        { role: 'user', content: [{ text: 'Review the preview.' }] },
        {
            role: 'assistant',
            content: [
                { toolUse: { toolUseId: 'preview', name: 'preview_artifact_officexml', input: {} } },
                { toolUse: { toolUseId: 'other', name: 'other_preview', input: {} } },
            ],
        },
        {
            role: 'user',
            content: [
                { toolResult: { toolUseId: 'preview', content: [{ text: 'Pages 1-8' }, { image }] } },
                { toolResult: { toolUseId: 'other', status: 'success', content: [{ image }] } },
            ],
        },
    ];
}

function expectImagesRelocated(request: ConverseRequest) {
    expect(request.messages?.[2]?.content).toEqual([
        { toolResult: { toolUseId: 'preview', content: [{ text: 'Pages 1-8' }] } },
        {
            toolResult: {
                toolUseId: 'other',
                status: 'success',
                content: [{ text: 'See the attached tool-result image(s).' }],
            },
        },
        { text: 'Image from tool result preview:' },
        { image },
        { text: 'Image from tool result other:' },
        { image },
    ]);
}

describe('Bedrock tool-result images', () => {
    it.each([MODEL, 'global.openai.gpt-7', 'mistral.pixtral-large-2502-v1:0'])(
        'preserves images, tool pairing and source history for %s',
        (model) => {
            const driver = new BedrockDriver({ region: 'us-east-1' });
            const original = messages();
            const before = structuredClone(original);
            const request = driver.preparePayload({ modelId: model, messages: original }, { model, tools });
            expectImagesRelocated(request);
            expect(original).toEqual(before);
            expect(driver.preparePayload(request, { model, tools }).messages).toEqual(request.messages);
        },
    );

    it.each(['us.anthropic.claude-sonnet-4-6', 'amazon.nova-pro-v1:0'])(
        'keeps supported nested images for %s',
        (model) => {
            const driver = new BedrockDriver({ region: 'us-east-1' });
            const original = messages();
            expect(driver.preparePayload({ modelId: model, messages: original }, { model, tools }).messages).toEqual(
                original,
            );
        },
    );

    it('preserves ordinary user images', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const original: Message[] = [{ role: 'user', content: [{ text: 'Describe this.' }, { image }] }];
        expect(driver.preparePayload({ modelId: MODEL, messages: original }, { model: MODEL }).messages).toEqual(
            original,
        );
    });

    it.each([
        { streaming: false, replay: false },
        { streaming: true, replay: false },
        { streaming: false, replay: true },
        { streaming: true, replay: true },
    ])('normalizes images at the API boundary ($streaming, replay=$replay)', async ({ streaming, replay }) => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const response: ConverseResponse = {
            output: { message: { role: 'assistant', content: [{ text: 'Reviewed.' }] } },
            stopReason: 'end_turn',
            usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
            metrics: { latencyMs: 1 },
        };
        const converse = vi.fn(async (_request: ConverseRequest) => response);
        const converseStream = vi.fn(async (_request: ConverseRequest) => ({
            stream: (async function* () {
                yield { messageStop: { stopReason: 'end_turn' as const } };
            })(),
        }));
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converse, converseStream, destroy: vi.fn() }),
        });
        const history = replay ? messages() : messages().slice(0, 2);
        // Match persisted binary storage, then append the current turn separately.
        const persisted = JSON.parse(
            JSON.stringify(history, (_key, value: unknown) =>
                value instanceof Uint8Array ? { _base64: Buffer.from(value).toString('base64') } : value,
            ),
        );
        const prompt: ConverseRequest = {
            modelId: MODEL,
            messages: replay
                ? [
                      { role: 'assistant', content: [{ text: 'Reviewed.' }] },
                      { role: 'user', content: [{ text: 'Review once more.' }] },
                  ]
                : messages().slice(2),
        };
        const options = { model: MODEL, tools, conversation: { messages: persisted } };
        if (streaming) {
            const stream = await driver.requestTextCompletionStream(prompt, options);
            for await (const _chunk of stream) {
                /* Drain to release the executor scope. */
            }
            expectImagesRelocated(converseStream.mock.calls[0][0]);
        } else {
            await driver.requestTextCompletion(prompt, options);
            expectImagesRelocated(converse.mock.calls[0][0]);
        }
    });
});
