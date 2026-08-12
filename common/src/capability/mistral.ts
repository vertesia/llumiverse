import type { ModelCapabilities } from '../types.js';

export interface MistralModelKnowledge {
    capabilities: ModelCapabilities;
    context_window?: number;
    reasoning_effort_levels?: readonly string[];
}

function datedVersionGte(model: string, family: string, target: number): boolean {
    const match = model.match(new RegExp(`${family}(\\d{4})(?:[-_.]|$)`));
    return !!match && Number(match[1]) >= target;
}

function semanticVersionGte(model: string, family: string, targetMajor: number, targetMinor = 0): boolean {
    const match = model.match(new RegExp(`${family}(\\d+)(?:[-.](\\d+))?(?:[-_.]|$)`));
    if (!match) return false;
    const major = Number(match[1]);
    const minor = Number(match[2] ?? 0);
    return major > targetMajor || (major === targetMajor && minor >= targetMinor);
}

function isMistralSmall4OrLater(model: string): boolean {
    return (
        model.includes('mistral-small-latest') ||
        datedVersionGte(model, 'mistral-small-', 2603) ||
        semanticVersionGte(model, 'mistral-small-', 4)
    );
}

function isMistralMedium35OrLater(model: string): boolean {
    return (
        model.includes('mistral-medium-latest') ||
        datedVersionGte(model, 'mistral-medium-', 2604) ||
        semanticVersionGte(model, 'mistral-medium-', 3, 5)
    );
}

export function isMistralAdjustableReasoningModel(model: string): boolean {
    const normalized = model.toLowerCase();
    return isMistralSmall4OrLater(normalized) || isMistralMedium35OrLater(normalized);
}

function hasMistralVision(model: string): boolean {
    return (
        model.includes('pixtral') ||
        model.includes('mistral-small-latest') ||
        datedVersionGte(model, 'mistral-small-', 2503) ||
        semanticVersionGte(model, 'mistral-small-', 3, 1) ||
        model.includes('mistral-medium-latest') ||
        datedVersionGte(model, 'mistral-medium-', 2508) ||
        semanticVersionGte(model, 'mistral-medium-', 3, 1) ||
        model.includes('mistral-large-latest') ||
        datedVersionGte(model, 'mistral-large-', 2512) ||
        semanticVersionGte(model, 'mistral-large-', 3) ||
        /ministral-(?:3|8|14)b-(?:latest|\d{4})(?:[-_.]|$)/.test(model)
    );
}

function getMistralContextWindow(model: string): number | undefined {
    if (model.includes('voxtral-small') || model.includes('voxtral-mini')) return 32_000;
    if (model.includes('magistral-')) {
        const version = model.match(/magistral-(?:small|medium)-(\d{4})(?:[-_.]|$)/)?.[1];
        return model.includes('latest') || (version && Number(version) >= 2509) ? 128_000 : 40_000;
    }
    if (
        isMistralSmall4OrLater(model) ||
        isMistralMedium35OrLater(model) ||
        model.includes('mistral-large-latest') ||
        datedVersionGte(model, 'mistral-large-', 2512) ||
        semanticVersionGte(model, 'mistral-large-', 3) ||
        /ministral-(?:3|8|14)b-(?:latest|\d{4})(?:[-_.]|$)/.test(model) ||
        model.includes('devstral-2') ||
        model.includes('leanstral')
    ) {
        return 256_000;
    }
    if (
        datedVersionGte(model, 'mistral-small-', 2503) ||
        semanticVersionGte(model, 'mistral-small-', 3, 1) ||
        datedVersionGte(model, 'mistral-medium-', 2508) ||
        semanticVersionGte(model, 'mistral-medium-', 3, 1) ||
        model.includes('mistral-nemo') ||
        model.includes('open-mistral-nemo') ||
        model.includes('codestral') ||
        model.includes('pixtral')
    ) {
        return 128_000;
    }
    if (/mistral-(?:small|medium|large)-24\d{2}/.test(model) || model.includes('open-mistral-7b')) {
        return 32_000;
    }
    return undefined;
}

export function getMistralModelKnowledge(model: string): MistralModelKnowledge {
    const normalized = model.toLowerCase();
    if (normalized.includes('moderation') || normalized.includes('shieldstral')) {
        return {
            capabilities: {
                input: { text: true },
                output: { text: true },
                tool_support: false,
                tool_support_streaming: false,
            },
        };
    }
    if (normalized.includes('mistral-ocr')) {
        return {
            capabilities: {
                input: { text: true, image: true },
                output: { text: true },
                tool_support: false,
                tool_support_streaming: false,
            },
        };
    }
    if (normalized.includes('tts')) {
        return {
            capabilities: {
                input: { text: true },
                output: { audio: true },
                tool_support: false,
                tool_support_streaming: false,
            },
        };
    }
    if (normalized.includes('voxtral-mini') && /(transcribe|realtime)/.test(normalized)) {
        return {
            capabilities: {
                input: { audio: true },
                output: { text: true },
                tool_support: false,
                tool_support_streaming: false,
            },
        };
    }

    const adjustableReasoning = isMistralAdjustableReasoningModel(normalized);
    const voxtralMini = normalized.includes('voxtral-mini');
    return {
        capabilities: {
            input: {
                text: true,
                image: hasMistralVision(normalized) || undefined,
                audio: normalized.includes('voxtral-small') || voxtralMini || undefined,
            },
            output: { text: true },
            tool_support: !voxtralMini,
            tool_support_streaming: !voxtralMini,
        },
        context_window: getMistralContextWindow(normalized),
        ...(adjustableReasoning && { reasoning_effort_levels: ['none', 'high'] }),
    };
}
