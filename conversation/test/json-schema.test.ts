import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, it } from 'vitest';
import { ConversationDocumentSchema, validateConversationDocument } from '../src/index.js';
import {
    ConversationContentBlockJsonSchema,
    ConversationDiagnosticJsonSchema,
    ConversationDocumentJsonSchema,
    ConversationGenerationJsonSchema,
    ConversationInspectionJsonSchema,
    ConversationTurnJsonSchema,
} from '../src/json-schema.js';
import { emptyDocument, generatedAgentTurn, importedGeneration, toolResultTurn } from './fixtures.js';

function compileDocumentSchema() {
    const ajv = new Ajv2020({ allErrors: true, strict: true });
    formatsPlugin.default(ajv);
    return ajv.compile(ConversationDocumentJsonSchema);
}

function expectSortedAndFrozen(value: unknown): void {
    if (Array.isArray(value)) {
        expect(Object.isFrozen(value)).toBe(true);
        for (const item of value) {
            expectSortedAndFrozen(item);
        }
        return;
    }
    if (value === null || typeof value !== 'object') {
        return;
    }
    expect(Object.isFrozen(value)).toBe(true);
    const keys = Object.keys(value);
    expect(keys).toEqual([...keys].sort((first, second) => (first < second ? -1 : first > second ? 1 : 0)));
    for (const item of Object.values(value)) {
        expectSortedAndFrozen(item);
    }
}

describe('generated JSON Schema', () => {
    it('exports deterministic, deeply frozen Draft 2020-12 schema objects', () => {
        for (const schema of [
            ConversationDocumentJsonSchema,
            ConversationTurnJsonSchema,
            ConversationContentBlockJsonSchema,
            ConversationGenerationJsonSchema,
            ConversationDiagnosticJsonSchema,
            ConversationInspectionJsonSchema,
        ]) {
            expect(schema.$schema).toBe('https://json-schema.org/draft/2020-12/schema');
            expect(String(schema.$id)).toContain('2026-09-11.foundation.1');
            expectSortedAndFrozen(schema);
        }
    });

    it('compiles in strict Ajv 2020 mode with formats', () => {
        const validate = compileDocumentSchema();
        expect(validate(emptyDocument())).toBe(true);
    });

    it('matches Zod on positive and adversarial shape fixtures', () => {
        const validImported = emptyDocument();
        validImported.generations.imported = importedGeneration('imported');

        const unknownField = { ...emptyDocument(), surprise: true };

        const missingGenerationId = emptyDocument();
        const agent = generatedAgentTurn('agent', 'generation');
        Reflect.deleteProperty(agent, 'generation_id');
        missingGenerationId.turns.push(agent);

        const multipleToolResults = emptyDocument();
        const result = toolResultTurn('result', 'call');
        multipleToolResults.turns.push({ ...result, blocks: [...result.blocks, ...result.blocks] });

        const fixtures: unknown[] = [
            emptyDocument(),
            validImported,
            unknownField,
            missingGenerationId,
            multipleToolResults,
        ];
        const validate = compileDocumentSchema();
        for (const fixture of fixtures) {
            expect(validate(fixture), JSON.stringify(validate.errors)).toBe(
                ConversationDocumentSchema.safeParse(fixture).success,
            );
        }
    });

    it('keeps graph validation separate from shape parity', () => {
        const dangling = emptyDocument();
        dangling.turns.push(generatedAgentTurn('agent', 'missing-generation'));
        const validate = compileDocumentSchema();

        expect(validate(dangling)).toBe(true);
        expect(ConversationDocumentSchema.safeParse(dangling).success).toBe(true);
        expect(validateConversationDocument(dangling).success).toBe(false);
    });
});
