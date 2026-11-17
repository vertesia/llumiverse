import {
    type DataSource,
    DECISION_INPUT_NOT_SUPPORTED,
    type DecisionAnswer,
    type DecisionQuestion,
    type DecisionsOptions,
    type DecisionsResult,
    LlumiverseError,
    PromptRole,
    type PromptSegment,
} from '@llumiverse/common';
import { readStreamAsBase64, readStreamAsString } from '@llumiverse/core';
import type OpenAI from 'openai';
import { openAIPromptUsage } from './usage.js';

type DecisionCreateParams = OpenAI.Decisions.DecisionCreateParams;
type DecisionInputItem = Exclude<DecisionCreateParams['input'], string>[number];
type DecisionInputPart = OpenAI.Decisions.DecisionInputPart;
type DecisionQuestionParam = DecisionCreateParams['questions'][number];
type OpenAIDecision = OpenAI.Decisions.Decision;

function unsupportedInput(provider: string, model: string, detail: string): LlumiverseError {
    return new LlumiverseError(
        `The ${provider} decisions endpoint does not accept ${detail}.`,
        false,
        { provider, model, operation: 'execute' },
        undefined,
        400,
        DECISION_INPUT_NOT_SUPPORTED,
    );
}

/**
 * The decisions endpoint takes text and images (URL or base64), but no PDF, audio, video or file id. Text files are
 * inlined. Anything else cannot be represented, so it fails here with a typed error the caller can route on, rather
 * than being dropped from the input.
 */
async function fileToDecisionPart(file: DataSource, provider: string, model: string): Promise<DecisionInputPart> {
    const mimeType = file.mime_type;
    if (!mimeType || mimeType.startsWith('image/')) {
        const base64 = await readStreamAsBase64(await file.getStream());
        return { type: 'input_image', image_url: `data:${mimeType || 'image/jpeg'};base64,${base64}` };
    }
    if (mimeType.startsWith('text/')) {
        return { type: 'input_text', text: await readStreamAsString(await file.getStream()) };
    }
    throw unsupportedInput(provider, model, `'${mimeType}' input (${file.name ?? 'unnamed file'})`);
}

/**
 * Converts prompt segments into decision input items, in the order the Responses formatter uses: system and safety
 * messages first, then the conversation. Negative and mask segments have no meaning for a decision and are skipped.
 */
export async function toOpenAIDecisionInput(
    segments: PromptSegment[],
    provider: string,
    model: string,
): Promise<DecisionInputItem[]> {
    const system: DecisionInputItem[] = [];
    const others: DecisionInputItem[] = [];

    for (const segment of segments) {
        if (segment.role === PromptRole.negative || segment.role === PromptRole.mask) continue;
        const parts: DecisionInputPart[] = [];
        for (const file of segment.files ?? []) parts.push(await fileToDecisionPart(file, provider, model));
        if (segment.content) parts.push({ type: 'input_text', text: segment.content });

        if (segment.role === PromptRole.system || segment.role === PromptRole.safety) {
            const text = textOnly(parts, provider, model, 'system');
            system.push({
                role: 'system',
                content: segment.role === PromptRole.safety ? `IMPORTANT: ${text}` : text,
            });
        } else if (segment.role === PromptRole.tool) {
            if (!segment.tool_use_id) throw new Error('Tool use id is required for tool messages');
            others.push({
                type: 'function_call_output',
                call_id: segment.tool_use_id,
                output: parts.length === 1 && parts[0].type === 'input_text' ? parts[0].text : toolParts(parts),
            });
        } else if (segment.role === PromptRole.assistant) {
            others.push({ role: 'assistant', content: textOnly(parts, provider, model, 'assistant') });
        } else {
            others.push({ role: 'user', content: parts });
        }
    }
    return [...system, ...others];
}

function textOnly(parts: DecisionInputPart[], provider: string, model: string, role: string): string {
    if (parts.some((part) => part.type !== 'input_text')) {
        throw unsupportedInput(provider, model, `files in ${role} messages`);
    }
    return parts.map((part) => (part.type === 'input_text' ? part.text : '')).join('\n');
}

function toolParts(
    parts: DecisionInputPart[],
): Array<OpenAI.Decisions.DecisionInputText | OpenAI.Decisions.DecisionInputImage> {
    return parts.filter(
        (part): part is OpenAI.Decisions.DecisionInputText | OpenAI.Decisions.DecisionInputImage =>
            part.type === 'input_text' || part.type === 'input_image',
    );
}

/** Llumiverse names the yes/no type `boolean`; OpenAI calls it `predicate`. */
export function toOpenAIDecisionQuestions(questions: DecisionQuestion[]): DecisionQuestionParam[] {
    return questions.map(toOpenAIDecisionQuestion);
}

function toOpenAIDecisionQuestion(question: DecisionQuestion): DecisionQuestionParam {
    switch (question.type) {
        case 'boolean':
            return { type: 'predicate', name: question.name, instructions: question.instructions };
        case 'choice':
            return {
                type: 'choice',
                name: question.name,
                instructions: question.instructions,
                choices: question.options.map((option) => ({
                    value: option.value,
                    ...(option.description !== undefined ? { description: option.description } : {}),
                })),
            };
        case 'score':
            return {
                type: 'score',
                name: question.name,
                instructions: question.instructions,
                levels: question.levels.map((level) => ({
                    label: level.label,
                    ...(level.description !== undefined ? { description: level.description } : {}),
                })),
            };
    }
}

/**
 * Maps OpenAI answers back to llumiverse answers. Answers come in question order and carry the question name; the
 * order is the fallback when a name is missing.
 */
export function fromOpenAIDecision(decision: OpenAIDecision, questions: DecisionQuestion[]): DecisionsResult {
    const answers = decision.answers.map((answer, index) =>
        fromOpenAIAnswer(answer, answer.name ?? questions[index]?.name ?? String(index)),
    );
    const usage = decision.usage;
    return {
        answers,
        model: decision.model,
        token_usage: usage
            ? {
                  ...openAIPromptUsage(
                      usage.input_tokens,
                      usage.input_tokens_details?.cached_tokens,
                      usage.input_tokens_details?.cache_write_tokens,
                  ),
                  result: usage.output_tokens,
                  total: usage.total_tokens,
              }
            : undefined,
    };
}

function fromOpenAIAnswer(answer: OpenAIDecision['answers'][number], name: string): DecisionAnswer {
    switch (answer.type) {
        case 'predicate':
            return { type: 'boolean', name, probability: answer.probability };
        case 'choice':
            return {
                type: 'choice',
                name,
                value: String(answer.choice),
                confidence: answer.confidence,
                probabilities: Object.fromEntries(
                    answer.probabilities.map((entry) => [String(entry.value), entry.probability]),
                ),
            };
        case 'score': {
            const level = answer.probabilities.find((entry) => entry.value === answer.score);
            return {
                type: 'score',
                name,
                score: answer.score,
                ...(level ? { label: level.label } : {}),
                confidence: answer.confidence,
                probabilities: Object.fromEntries(
                    answer.probabilities.map((entry) => [entry.label, entry.probability]),
                ),
            };
        }
        case 'refusal':
            return { type: 'refusal', name };
    }
}

export async function buildOpenAIDecisionRequest(
    options: DecisionsOptions,
    provider: string,
): Promise<DecisionCreateParams> {
    return {
        model: options.model,
        input: await toOpenAIDecisionInput(options.input, provider, options.model),
        questions: toOpenAIDecisionQuestions(options.questions),
    };
}
