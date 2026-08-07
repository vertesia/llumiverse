import { describe, expect, it } from 'vitest';
import { getModelCapabilities, isEmbeddingModel, supportsToolUse } from './capability.js';
import { Providers } from './types.js';

describe('embedding model classification', () => {
    it.each([
        [{ id: 'text-embedding-3-small' }, 'openai'],
        [{ id: 'text-embedding-ada-002' }, 'azure_openai'],
        [{ id: 'mistral-embed' }, 'mistralai'],
        [{ id: 'embedding-model', output_modalities: ['vectors'] }, 'xai'],
        [{ id: 'chat-model', type: 'embedding' }, 'togetherai'],
    ] as const)('recognizes an embedding listing: %s', (model, provider) => {
        expect(isEmbeddingModel(model, provider)).toBe(true);
    });

    it('does not classify normal inference models as embeddings', () => {
        expect(isEmbeddingModel({ id: 'gpt-5.6-sol' }, 'openai')).toBe(false);
        expect(isEmbeddingModel({ id: 'grok-4.3', output_modalities: ['text'] }, 'xai')).toBe(false);
    });

    it('filters non-standard audio and moderation models from inference listings', () => {
        expect(isEmbeddingModel({ id: 'whisper-large-v3' }, Providers.groq)).toBe(true);
        expect(isEmbeddingModel({ id: 'canopylabs/orpheus-v1-english' }, Providers.groq)).toBe(true);
        expect(isEmbeddingModel({ id: 'meta-llama/llama-prompt-guard-2-86m' }, Providers.groq)).toBe(true);
        expect(isEmbeddingModel({ id: 'llama-3.3-70b-versatile' }, Providers.groq)).toBe(false);
    });
});

describe('xAI Grok tool capabilities', () => {
    it.each(['grok-2', 'grok-3', 'grok-4', 'grok-4-fast-reasoning'])(
        'enables tool use for %s via Providers.xai without setting tool_support_streaming',
        (model) => {
            const caps = getModelCapabilities(model, Providers.xai);
            expect(caps.tool_support).toBe(true);
            expect(caps.tool_support_streaming).toBeUndefined();
            // Streaming agents must still attach tools when the flag is omitted
            expect(supportsToolUse(model, Providers.xai, true)).toBe(true);
            expect(supportsToolUse(model, Providers.xai, false)).toBe(true);
        },
    );

    it('enables streaming tool use when provider is omitted for grok-* models', () => {
        const caps = getModelCapabilities('grok-3');
        expect(caps.tool_support).toBe(true);
        expect(caps.tool_support_streaming).toBeUndefined();
        expect(supportsToolUse('grok-3', undefined, true)).toBe(true);
    });
});

describe('supportsToolUse streaming default', () => {
    it('falls back to tool_support when tool_support_streaming is unset', () => {
        // Vertex Grok records tool_support but not tool_support_streaming
        expect(getModelCapabilities('grok-3', Providers.vertexai).tool_support).toBe(true);
        expect(getModelCapabilities('grok-3', Providers.vertexai).tool_support_streaming).toBeUndefined();
        expect(supportsToolUse('grok-3', Providers.vertexai, true)).toBe(true);
    });

    it('honors explicit tool_support_streaming: false', () => {
        // Bedrock runtime Llama has tools but not streaming tools
        const caps = getModelCapabilities('meta.llama3-1-70b-instruct-v1:0', Providers.bedrock);
        expect(caps.tool_support).toBe(true);
        expect(caps.tool_support_streaming).toBe(false);
        expect(supportsToolUse('meta.llama3-1-70b-instruct-v1:0', Providers.bedrock, true)).toBe(false);
        expect(supportsToolUse('meta.llama3-1-70b-instruct-v1:0', Providers.bedrock, false)).toBe(true);
    });
});
