import { GenerateContentResponse, GoogleGenAI } from '@google/genai';
import { type DataSource, type ExecutionOptions, PromptRole } from '@llumiverse/core';
import OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { AnthropicDriver } from '../anthropic/index.js';
import { AzureFoundryDriver } from '../azure/azure_foundry.js';
import { formatConversePrompt } from '../bedrock/converse.js';
import { BedrockMantleDriver } from '../bedrock-mantle/index.js';
import { OpenAIDriver } from '../openai/openai.js';
import { OpenAIChatCompletionsDriver } from '../openai/openai_chat_completions.js';
import { VertexAIDriver } from '../vertexai/index.js';

vi.mock('@aws/bedrock-token-generator', () => ({ getTokenProvider: vi.fn(() => async () => 'test') }));
const bytes = new Uint8Array([1, 2, 3, 4]);
const base64 = Buffer.from(bytes).toString('base64');
function file(uri = 'gs://bucket/recording.wav'): DataSource {
    return {
        name: 'recording.wav',
        mime_type: 'audio/wav',
        getURI: vi.fn(async () => uri),
        getURL: vi.fn(async () => 'https://example.test/recording.wav'),
        getStream: vi.fn(async () => new Blob([bytes]).stream()),
    };
}
const prompt = [{ role: PromptRole.user, content: 'Hello.' }];
const store = vi.fn<NonNullable<ExecutionOptions['store_audio']>>(async (stream) => {
    expect(new Uint8Array(await new Response(stream).arrayBuffer())).toEqual(bytes);
    return 'gs://bucket/speech.pcm';
});
function chatResponse(model: string) {
    return {
        id: 'test',
        object: 'chat.completion' as const,
        created: 1,
        model,
        choices: [
            {
                index: 0,
                message: { role: 'assistant' as const, content: 'A greeting.', refusal: null },
                finish_reason: 'stop' as const,
                logprobs: null,
            },
        ],
    };
}

describe('primary provider file audio', () => {
    it('routes Foundry speech using the deployment name and its SDK client', async () => {
        const service = new OpenAI({ apiKey: 'test' });
        const create = vi.spyOn(service.audio.speech, 'create').mockResolvedValue(new Response(bytes));
        const driver = new AzureFoundryDriver({
            endpoint: 'https://example.test',
            azureADTokenProvider: { getToken: async () => ({ token: 'test', expiresOnTimestamp: Date.now() + 60000 }) },
        });
        driver.service = {
            deployments: { get: async () => ({ modelPublisher: 'OpenAI' }) },
            getOpenAIClient: () => service,
        } as unknown as AzureFoundryDriver['service'];
        const result = await driver.execute(prompt, {
            model: 'speech-deployment::gpt-4o-mini-tts',
            store_audio: store,
        });
        expect(create).toHaveBeenCalledWith(expect.objectContaining({ model: 'speech-deployment' }), expect.anything());
        expect(result.result[0].type).toBe('audio');
        expect(result.prompt).toEqual([]);
    });

    it.each(['openai', 'compatible'] as const)(
        'uses bounded Chat audio for %s without retaining bytes',
        async (provider) => {
            const driver =
                provider === 'openai'
                    ? new OpenAIDriver({ apiKey: 'test' })
                    : new OpenAIChatCompletionsDriver({ apiKey: 'test', endpoint: 'https://example.test/v1' });
            const create = vi
                .spyOn(driver.service.chat.completions, 'create')
                .mockResolvedValue(chatResponse('gpt-audio'));
            const result = await driver.execute([{ role: PromptRole.user, content: 'Describe', files: [file()] }], {
                model: 'gpt-audio',
                include_original_response: true,
            });
            expect(create).toHaveBeenCalledWith(
                expect.objectContaining({
                    modalities: ['text'],
                    messages: [
                        {
                            role: 'user',
                            content: [
                                { type: 'text', text: 'Describe' },
                                { type: 'input_audio', input_audio: { data: base64, format: 'wav' } },
                            ],
                        },
                    ],
                }),
                expect.anything(),
            );
            expect(JSON.stringify(result)).not.toContain(base64);
            expect(result.conversation).toBeUndefined();
        },
    );

    it('requests diarized JSON and retains speaker segments', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        driver.service = driver.service.withOptions({
            fetch: async (_url, init) => {
                const body = await new Response(init?.body).text();
                expect(body).toContain('diarized_json');
                expect(body).toContain('chunking_strategy');
                return Response.json({
                    text: 'Hello',
                    segments: [{ id: '0', speaker: 'A', start: 0, end: 1, text: 'Hello' }],
                });
            },
        });
        const result = await driver.execute([{ role: PromptRole.user, content: '', files: [file()] }], {
            model: 'gpt-4o-transcribe-diarize',
        });
        expect(result.result).toEqual([
            { type: 'text', value: 'Hello' },
            { type: 'json', value: { segments: [{ id: '0', speaker: 'A', start: 0, end: 1, text: 'Hello' }] } },
        ]);
    });

    it('stores Vertex PCM and delivers the fallback completion without inline audio', async () => {
        const driver = new VertexAIDriver({ project: 'test', region: 'global' });
        const client = new GoogleGenAI({ apiKey: 'test' });
        const response = new GenerateContentResponse();
        response.candidates = [
            { content: { parts: [{ inlineData: { mimeType: 'audio/L16;codec=pcm;rate=24000', data: base64 } }] } },
        ];
        const generate = vi.spyOn(client.models, 'generateContent').mockResolvedValue(response);
        vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue(client);
        const stream = await driver.stream(prompt, { model: 'gemini-3.1-flash-tts-preview', store_audio: store });
        for await (const _chunk of stream) {
            /* consume finite completion */
        }
        expect(generate).toHaveBeenCalledWith(
            expect.objectContaining({ config: expect.objectContaining({ responseModalities: ['AUDIO'] }) }),
        );
        expect(stream.completion?.result[0]).toMatchObject({
            type: 'audio',
            value: 'gs://bucket/speech.pcm',
            sample_rate: 24000,
            channels: 1,
            byte_order: 'little',
        });
        expect(JSON.stringify(stream.completion)).not.toContain(base64);
        expect(stream.completion?.conversation).toBeUndefined();
    });

    it('passes Vertex GCS transcription as fileData and returns transcript metadata', async () => {
        const driver = new VertexAIDriver({ project: 'test', region: 'global' });
        const client = new GoogleGenAI({ apiKey: 'test' });
        const response = new GenerateContentResponse();
        response.candidates = [
            {
                content: {
                    parts: [
                        {
                            text: 'Hello',
                            audioTranscription: {
                                text: 'Hello',
                                speakerLabel: 'A',
                                words: [{ word: 'Hello', startOffset: '0s', endOffset: '1s' }],
                            },
                        },
                    ],
                },
            },
        ];
        const generate = vi.spyOn(client.models, 'generateContent').mockResolvedValue(response);
        vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue(client);
        const audio = file();
        const result = await driver.execute([{ role: PromptRole.user, content: '', files: [audio] }], {
            model: 'gemini-3.5-transcribe-preview',
            model_options: { _option_id: 'vertexai-gemini', transcription_diarization: true },
        });
        expect(generate).toHaveBeenCalledWith(
            expect.objectContaining({
                contents: [
                    {
                        role: 'user',
                        parts: [{ fileData: { fileUri: 'gs://bucket/recording.wav', mimeType: 'audio/wav' } }],
                    },
                ],
                config: expect.objectContaining({
                    audioTranscriptionConfig: expect.objectContaining({ diarization: true }),
                }),
            }),
        );
        expect(audio.getStream).not.toHaveBeenCalled();
        expect(audio.getURL).not.toHaveBeenCalled();
        expect(result.result[0]).toEqual({ type: 'text', value: 'Hello' });
        expect(result.result).toHaveLength(2);
    });

    it('preserves S3 audio references in Converse without reading the file', async () => {
        const audio = file('s3://bucket/recording.wav');
        const result = await formatConversePrompt([{ role: PromptRole.user, content: 'Describe', files: [audio] }], {
            model: 'mistral.voxtral-small-24b-2507',
        });
        expect(JSON.stringify(result)).toContain('s3Location');
        expect(JSON.stringify(result)).toContain('s3://bucket/recording.wav');
        expect(audio.getStream).not.toHaveBeenCalled();
        expect(audio.getURL).not.toHaveBeenCalled();
    });

    it('sends Mantle Voxtral input_audio through Chat and strips persisted payloads', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });
        const create = vi
            .spyOn(driver.service.chat.completions, 'create')
            .mockResolvedValue(chatResponse('mistral.voxtral-small-24b-2507'));
        const result = await driver.execute([{ role: PromptRole.user, content: 'Describe', files: [file()] }], {
            model: 'mistral.voxtral-small-24b-2507',
            include_original_response: true,
        });
        expect(JSON.stringify(create.mock.calls[0][0])).toContain(base64);
        expect(JSON.stringify(result)).not.toContain(base64);
        expect(result.result[0]).toMatchObject({ type: 'text', value: 'A greeting.' });
    });

    it('rejects native Claude audio explicitly before inference', async () => {
        const driver = new AnthropicDriver({ apiKey: 'test' });
        const create = vi.spyOn(driver.client.messages, 'create');
        await expect(
            driver.execute([{ role: PromptRole.user, content: 'Describe', files: [file()] }], {
                model: 'claude-sonnet-4-6',
            }),
        ).rejects.toThrow('does not support audio input');
        expect(create).not.toHaveBeenCalled();
    });
});
