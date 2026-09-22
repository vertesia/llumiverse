import type { ExecutionResponse } from '@llumiverse/common';
import { describe, expect, it } from 'vitest';
import { stripAudioFromCompletion, stripAudioPayloads } from './conversation-utils.js';

const applicationData = {
    audio: { data: 'customer metadata', source: { bytes: [1, 2] } },
    inlineData: { mimeType: 123 },
    type: 'input_audio',
    input_audio: { data: 'application value' },
    messages: [{ role: 'user', content: [{ type: 'input_audio', input_audio: { data: 'business data' } }] }],
};

describe('audio history boundaries', () => {
    it('never visits completion results, tool arguments, errors or diagnostics', () => {
        const completion: ExecutionResponse = {
            prompt: { messages: [{ role: 'user', content: 'hello' }], schema: applicationData },
            result: [{ type: 'json', value: applicationData }],
            tool_use: [{ id: 'call', tool_name: 'save', tool_input: applicationData }],
            error: { code: 'validation_error', message: 'example', data: [{ type: 'json', value: applicationData }] },
        };
        const saved = stripAudioFromCompletion(completion);
        expect(saved).toEqual(completion);
        expect(saved.result).toBe(completion.result);
        expect(saved.tool_use).toBe(completion.tool_use);
        expect(saved.error).toBe(completion.error);
        expect(saved.prompt).toBe(completion.prompt);
    });

    it('removes media only in provider blocks while preserving tool payloads and durable references', () => {
        const opaqueBlocks = [
            { type: 'tool_use', id: 'call', name: 'save', input: applicationData },
            { type: 'tool_result', tool_use_id: 'call', content: applicationData },
            { functionCall: { name: 'save', args: applicationData } },
            { functionResponse: { name: 'save', response: applicationData } },
            { toolUse: { toolUseId: 'call', name: 'save', input: applicationData } },
            { toolResult: { toolUseId: 'call', content: [{ json: applicationData }] } },
            { inlineData: { mimeType: 123 } },
            { inlineData: { mimeType: 'image/png', data: 'image-data' } },
            { fileData: { fileUri: 'gs://bucket/recording.wav', mimeType: 'audio/wav' } },
            { audio: { source: { s3Location: { uri: 's3://bucket/recording.wav' } }, format: 'wav' } },
        ];
        for (const field of ['content', 'parts']) {
            const messages = [
                {
                    role: 'user',
                    [field]: [
                        { type: 'input_audio', input_audio: { data: 'secret-audio', format: 'wav' } },
                        { audio: { source: { bytes: new Uint8Array([1, 2]) }, format: 'wav' } },
                        { inlineData: { mimeType: 'audio/wav', data: 'secret-audio' } },
                        ...opaqueBlocks,
                    ],
                },
            ];
            for (const envelope of [messages, { messages }, { contents: messages }, { _arrayConversation: messages }]) {
                const saved = stripAudioPayloads(envelope);
                expect(JSON.stringify(saved)).not.toContain('secret-audio');
                for (const block of opaqueBlocks) expect(JSON.stringify(saved)).toContain(JSON.stringify(block));
                expect(JSON.stringify(envelope)).toContain('secret-audio');
            }
        }
    });

    it('preserves non-audio SDK instances and opaque extensions by identity', () => {
        class Response {
            candidates = [{ content: { role: 'model', parts: [{ functionCall: { args: applicationData } }] } }];
            custom = applicationData;
            get text() {
                return 'hello';
            }
        }
        const response = new Response();
        expect(stripAudioPayloads(response)).toBe(response);
        expect(stripAudioPayloads(response).text).toBe('hello');
    });

    it('cleans actual response audio without traversing sibling provider data', () => {
        const gemini = {
            candidates: [{ content: { parts: [{ inlineData: { mimeType: 'audio/wav', data: 'secret-audio' } }] } }],
            custom: applicationData,
        };
        const chat = {
            choices: [
                {
                    message: {
                        role: 'assistant',
                        content: 'Transcript',
                        audio: { data: 'secret-audio', transcript: 'Transcript', id: 'audio-id' },
                        tool_calls: [{ function: { arguments: JSON.stringify(applicationData) } }],
                    },
                },
            ],
        };
        const saved = stripAudioPayloads(chat);
        expect(JSON.stringify(saved)).not.toContain('secret-audio');
        expect(saved.choices[0].message.audio).toEqual({ transcript: 'Transcript', id: 'audio-id' });
        expect(saved.choices[0].message.tool_calls).toBe(chat.choices[0].message.tool_calls);
        expect(stripAudioPayloads(gemini).custom).toBe(applicationData);
        expect(JSON.stringify(stripAudioPayloads(gemini))).not.toContain('secret-audio');
    });
});
