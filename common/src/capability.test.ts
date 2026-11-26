import { describe, expect, it } from 'vitest';
import {
    getModelCapabilities,
    isDedicatedInferenceModel,
    isEmbeddingModel,
    supportsDecisions,
    supportsToolUse,
} from './capability.js';
import { Providers } from './types.js';

describe('dedicated inference model classification', () => {
    it.each(['gpt-image-1', 'text-embedding-3-small', 'whisper-1', 'sora-2'])(
        'recognizes %s as a dedicated endpoint model',
        (model) => {
            expect(isDedicatedInferenceModel(model, Providers.openai)).toBe(true);
        },
    );

    it('keeps unknown future models in normal inference listings', () => {
        expect(isDedicatedInferenceModel('future-inference-model-v1', Providers.openai)).toBe(false);
    });
});

describe('embedding model classification', () => {
    it.each([
        [{ id: 'text-embedding-3-small' }, Providers.openai],
        [{ id: 'text-embedding-ada-002' }, Providers.azure_openai],
        [{ id: 'mistral-embed' }, Providers.mistralai],
        [{ id: 'embedding-model', output_modalities: ['vectors'] }, Providers.xai],
        [{ id: 'chat-model', type: 'embedding' }, Providers.togetherai],
    ] as const)('recognizes an embedding listing: %s', (model, provider) => {
        expect(isEmbeddingModel(model, provider)).toBe(true);
    });

    it('does not classify normal inference models as embeddings', () => {
        expect(isEmbeddingModel({ id: 'gpt-5.6-sol' }, Providers.openai)).toBe(false);
        expect(isEmbeddingModel({ id: 'grok-4.3', output_modalities: ['text'] }, Providers.xai)).toBe(false);
    });

    it('does not hide non-embedding models when capability metadata is incomplete', () => {
        expect(isEmbeddingModel({ id: 'whisper-large-v3' }, Providers.groq)).toBe(false);
        expect(isEmbeddingModel({ id: 'canopylabs/orpheus-v1-english' }, Providers.groq)).toBe(false);
        expect(isEmbeddingModel({ id: 'meta-llama/llama-prompt-guard-2-86m' }, Providers.groq)).toBe(false);
        expect(isEmbeddingModel({ id: 'llama-3.3-70b-versatile' }, Providers.groq)).toBe(false);
        expect(isEmbeddingModel({ id: 'gpt-image-1' }, Providers.openai)).toBe(false);
        expect(isEmbeddingModel({ id: 'amazon.titan-image-generator-v3' }, Providers.bedrock)).toBe(false);
        expect(isEmbeddingModel({ id: 'amazon.nova-reel-v1' }, Providers.bedrock)).toBe(false);
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

    it('carries verified image input support into current and future Grok 4 models', () => {
        expect(getModelCapabilities('grok-4.20', Providers.xai).input.image).toBe(true);
        expect(getModelCapabilities('grok-4.5', Providers.xai).input.image).toBe(true);
    });

    it('preserves supported audio and video inputs while masking unsupported outputs', () => {
        const caps = getModelCapabilities('gemini-4.0-flash', Providers.vertexai);
        expect(caps.input.audio).toBe(true);
        expect(caps.input.video).toBe(true);
        expect(caps.output.audio).toBe(false);
        expect(caps.output.video).toBe(false);
    });
});

describe('Gemini Omni video capabilities', () => {
    it.each([
        'gemini-omni-flash-preview',
        'gemini-omni-1.1-flash-preview',
        'gemini-omni-1.2-flash',
        'publishers/google/models/gemini-omni-1.3-flash',
    ])('exposes the executable media capabilities for %s', (model) => {
        expect(getModelCapabilities(model, Providers.vertexai)).toEqual({
            input: { text: true, image: true, video: true, audio: false, embed: false },
            output: { text: true, image: false, video: true, audio: false, embed: false },
            tool_support: false,
            tool_support_streaming: false,
        });
    });
});

describe('transport input capabilities', () => {
    it('advertises only the inputs the Chat Completions drivers can send', () => {
        // The source model takes all four inputs; OpenRouter's request conversion sends text and images only.
        expect(getModelCapabilities('qwen/qwen3.8-omni-flash', Providers.openrouter).input).toMatchObject({
            text: true,
            image: true,
            video: false,
            audio: false,
        });
        expect(getModelCapabilities('qwen3.8-omni-flash', Providers.openai_compatible).input).toMatchObject({
            image: true,
            video: false,
            audio: true,
        });
        expect(getModelCapabilities('google/gemma-4-31b-it', Providers.togetherai).input.video).toBe(false);
        expect(getModelCapabilities('google/gemini-3.5-flash', Providers.openrouter).input).toMatchObject({
            video: false,
            audio: false,
        });
    });

    it('keeps video and audio for drivers that send them', () => {
        expect(getModelCapabilities('gemini-3.5-flash', Providers.vertexai).input).toMatchObject({
            video: true,
            audio: true,
        });
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

    it('leaves genuinely unknown Bedrock tool support unset', () => {
        const caps = getModelCapabilities('future-provider.unknown-chat-v1', Providers.bedrock);
        expect(caps.tool_support).toBeUndefined();
        expect(caps.tool_support_streaming).toBeUndefined();
    });

    it.each([
        ['my-finetune-v3', Providers.openai_compatible],
        ['unsloth/some-new-model-GGUF', Providers.openai_compatible],
        ['future-provider.unknown-chat-v1', Providers.bedrock],
    ])('sends tools for unknown model %s on %s', (model, provider) => {
        expect(getModelCapabilities(model, provider).tool_support).toBeUndefined();
        expect(supportsToolUse(model, provider)).toBe(true);
        expect(supportsToolUse(model, provider, true)).toBe(true);
    });

    it.each([
        ['text-embedding-3-small', Providers.openai],
        ['gpt-image-1', Providers.openai],
        ['whisper-1', Providers.openai_compatible],
        ['tts-1', Providers.openai_compatible],
    ])('keeps tools off for %s on %s, which the directory marks as not tool-capable', (model, provider) => {
        expect(supportsToolUse(model, provider)).toBe(false);
        expect(supportsToolUse(model, provider, true)).toBe(false);
    });
});

describe('decision support', () => {
    it.each([
        ['gpt-6-luna', true],
        ['gpt-6-luna-2026-10-22', true],
        ['gpt-7-luna', true],
        ['gpt-6.5-luna', true],
        ['gpt-5.6-luna', false],
        ['gpt-6-astra', false],
        ['gpt-6', false],
    ])('%s on OpenAI: %s', (model, expected) => {
        expect(supportsDecisions(model, Providers.openai)).toBe(expected);
    });

    it('is scoped to the provider that serves the endpoint', () => {
        expect(supportsDecisions('gpt-6-luna', Providers.azure_openai)).toBe(false);
        expect(supportsDecisions('openai/gpt-6-luna', Providers.openrouter)).toBe(false);
    });
});
