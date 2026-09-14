import { describe, expect, it } from 'vitest';
import { stripAudioPayloads } from './conversation-utils.js';

describe('audio history boundaries', () => {
    it('removes inline audio across protocol shapes without mutating requests or removing durable references', () => {
        class Response {
            candidates = [{ content: { parts: [{ inlineData: { mimeType: 'audio/wav', data: 'secret-audio' } }] } }];
        }
        const request = {
            original_response: new Response(),
            conversation: [
                { type: 'input_audio', input_audio: { data: 'secret-audio', format: 'wav' } },
                { audio: { source: { bytes: new Uint8Array([1, 2, 3]) }, format: 'wav' } },
                { role: 'assistant', content: 'Transcript', audio: { data: 'secret-audio', transcript: 'Transcript' } },
                { fileData: { fileUri: 'gs://bucket/recording.wav', mimeType: 'audio/wav' } },
                { audio: { source: { s3Location: { uri: 's3://bucket/recording.wav' } }, format: 'wav' } },
                { inlineData: { mimeType: 'image/png', data: 'image-data' } },
            ],
            result: [{ type: 'audio', value: 'gs://bucket/speech.wav', mime_type: 'audio/wav' }],
        };
        const saved = stripAudioPayloads(request);
        expect(JSON.stringify(saved)).not.toContain('secret-audio');
        expect(JSON.stringify(saved)).not.toContain('bytes');
        expect(JSON.stringify(saved)).toContain('s3://bucket/recording.wav');
        expect(JSON.stringify(saved)).toContain('gs://bucket/recording.wav');
        expect(JSON.stringify(saved)).toContain('image-data');
        expect(saved.result).toEqual(request.result);
        expect(JSON.stringify(request)).toContain('secret-audio');
    });
});
