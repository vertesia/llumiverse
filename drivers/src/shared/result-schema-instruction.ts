import { RESULT_SCHEMA_CLOSE_TAG, RESULT_SCHEMA_OPEN_TAG } from '@llumiverse/core';

/**
 * The prompt copy of a result schema, which drivers add when native structured output is missing or unreliable.
 * One builder keeps the text identical across protocols, so a copy already in a conversation can be recognised.
 */
const RESULT_SCHEMA_INSTRUCTION_PREFIX = `IMPORTANT: only answer using JSON, and respecting the schema included below, between the ${RESULT_SCHEMA_OPEN_TAG} tags.`;

export function buildResultSchemaInstruction(schema: unknown): string {
    return `${RESULT_SCHEMA_INSTRUCTION_PREFIX} ${RESULT_SCHEMA_OPEN_TAG}${JSON.stringify(schema)}${RESULT_SCHEMA_CLOSE_TAG}`;
}

export function isResultSchemaInstruction(text: unknown): text is string {
    return typeof text === 'string' && text.startsWith(RESULT_SCHEMA_INSTRUCTION_PREFIX);
}

/**
 * Merges a new prompt into a conversation without storing one schema copy per turn.
 *
 * Copies already in the history stay where they are, so the cached prompt prefix is unchanged: an incoming copy
 * identical to a stored one is dropped. A copy with a different schema replaces the stored ones, since two schemas
 * would contradict each other. `instructionText` returns the instruction text of a message that is a schema copy
 * eligible for deduplication, and undefined otherwise (for example a suffix copy, which is left alone).
 */
export function mergeResultSchemaInstructions<T>(
    history: readonly T[],
    incoming: readonly T[],
    instructionText: (message: T) => string | undefined,
): T[] {
    const incomingTexts = incoming.map(instructionText);
    const incomingText = incomingTexts.find((text) => text !== undefined);
    if (incomingText === undefined) return [...history, ...incoming];

    const stored = history.map(instructionText).filter((text) => text !== undefined);
    if (stored.includes(incomingText)) {
        return [...history, ...incoming.filter((_message, index) => incomingTexts[index] === undefined)];
    }
    const keptHistory = stored.length ? history.filter((message) => instructionText(message) === undefined) : history;
    return [...keptHistory, ...incoming];
}
