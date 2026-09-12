import { describe, expect, it } from 'vitest';
import {
    ConversationTurnSchema,
    createTextBlock,
    parseConversationDocument,
    ToolTurnSchema,
    validateConversationDocument,
} from '../src/index.js';
import {
    emptyDocument,
    generatedAgentTurn,
    importedGeneration,
    RECORDED_AT,
    toolCallBlock,
    toolResultTurn,
    userTurn,
} from './fixtures.js';

const provenance = (method: 'reported' | 'derived' = 'reported') => ({
    method,
    accounting_basis: 'provider-v1',
});

describe('conversation shape validation', () => {
    it('rejects unknown fields on strict objects', () => {
        const input = { ...emptyDocument(), unknown_field: true };
        const result = validateConversationDocument(input);
        expect(result.success).toBe(false);
        expect(result.diagnostics[0]).toMatchObject({ code: 'SCHEMA_INVALID', stage: 'schema' });
    });

    it('enforces generated agent references in the emitted union shape', () => {
        const turn = generatedAgentTurn('agent', 'generation');
        Reflect.deleteProperty(turn, 'generation_id');
        expect(ConversationTurnSchema.safeParse(turn).success).toBe(false);
    });

    it('permits imported generations without invented model metadata', () => {
        const document = emptyDocument();
        document.generations.imported = importedGeneration('imported');
        expect(validateConversationDocument(document)).toMatchObject({ success: true });
    });

    it('requires one result per tool turn and excludes nested calls and results', () => {
        const turn = toolResultTurn('result-turn', 'call');
        expect(ToolTurnSchema.safeParse({ ...turn, blocks: [...turn.blocks, ...turn.blocks] }).success).toBe(false);
        expect(
            ToolTurnSchema.safeParse({
                ...turn,
                blocks: [{ ...turn.blocks[0], content: [toolCallBlock('nested-call', 'nested')] }],
            }).success,
        ).toBe(false);
        expect(
            ToolTurnSchema.safeParse({
                ...turn,
                blocks: [{ ...turn.blocks[0], content: [{ ...turn.blocks[0], id: 'nested-result' }] }],
            }).success,
        ).toBe(false);
    });

    it('rejects reserved __proto__ input rather than letting Zod rewrite it', () => {
        const document = emptyDocument();
        document.metadata = JSON.parse('{"__proto__":{"invalid":true}}');
        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics[0]?.code).toBe('JSON_RESERVED_PROPERTY_KEY');
    });

    it('bounds schema diagnostics for enormous unknown keys', () => {
        const result = validateConversationDocument({ ...emptyDocument(), ['x'.repeat(100_000)]: true });
        expect(result.success).toBe(false);
        expect(JSON.stringify(result.diagnostics).length).toBeLessThan(5_000);
    });
});

describe('identity and reference validation', () => {
    it('accepts valid own map entries whose IDs resemble Object.prototype names', () => {
        const document = emptyDocument();
        Reflect.set(document.tool_definitions, 'constructor', {
            id: 'constructor',
            name: 'read',
            version: '1',
            input_schema: true,
        });
        document.context.active_tool_definition_ids = ['constructor'];

        const result = validateConversationDocument(document);
        expect(result.success).toBe(true);
        if (result.success) {
            expect(Object.hasOwn(result.data.tool_definitions, 'constructor')).toBe(true);
        }
    });

    it.each(['constructor', 'toString', 'hasOwnProperty'])(
        'does not resolve absent %s through Object.prototype',
        (id) => {
            const document = emptyDocument();
            document.turns.push(generatedAgentTurn('agent', id));
            const result = validateConversationDocument(document);
            expect(result.success).toBe(false);
            expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'REFERENCE_NOT_FOUND')).toBe(true);
        },
    );

    it('rejects duplicate IDs across entity kinds', () => {
        const document = emptyDocument();
        document.turns.push(userTurn('same', 'same'));
        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'DUPLICATE_ID')).toBe(true);
    });

    it('rejects dangling media references', () => {
        const document = emptyDocument();
        const turn = userTurn('user');
        turn.blocks = [{ id: 'image', type: 'image', asset_id: 'missing' }];
        document.turns.push(turn);
        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'REFERENCE_NOT_FOUND')).toBe(true);
    });
});

describe('usage accounting', () => {
    it('requires canonical totals to equal a safe input plus output sum', () => {
        const document = emptyDocument();
        const generation = importedGeneration('generation');
        generation.usage = {
            input_tokens: 5,
            output_tokens: 7,
            total_tokens: 99,
            accounting_provenance: {
                input_tokens: provenance(),
                output_tokens: provenance(),
                total_tokens: provenance(),
            },
        };
        document.generations.generation = generation;

        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'USAGE_TOTAL_INVALID')).toBe(true);
    });

    it('requires provenance for every normalized measurement without fabricating defaults', () => {
        const document = emptyDocument();
        const generation = importedGeneration('generation');
        generation.usage = { input_tokens: 5 };
        document.generations.generation = generation;

        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'ACCOUNTING_PROVENANCE_MISMATCH')).toBe(
            true,
        );
    });

    it('rejects derived new-input usage without a complete disjoint partition', () => {
        const document = emptyDocument();
        const generation = importedGeneration('generation');
        generation.usage = {
            input_new_tokens: 25,
            accounting_provenance: { input_new_tokens: provenance('derived') },
        };
        document.generations.generation = generation;

        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'INPUT_PARTITION_INVALID')).toBe(true);
    });

    it('rejects a canonical total across incompatible accounting bases', () => {
        const document = emptyDocument();
        const generation = importedGeneration('generation');
        generation.usage = {
            input_tokens: 5,
            output_tokens: 7,
            total_tokens: 12,
            accounting_provenance: {
                input_tokens: { method: 'reported', accounting_basis: 'input-basis' },
                output_tokens: { method: 'reported', accounting_basis: 'output-basis' },
                total_tokens: { method: 'derived', accounting_basis: 'input-basis' },
            },
        };
        document.generations.generation = generation;

        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'USAGE_TOTAL_INVALID')).toBe(true);
    });

    it('rejects an empty reported-usage collection', () => {
        const document = emptyDocument();
        const generation = importedGeneration('generation');
        generation.usage = { reported_usage: [] };
        document.generations.generation = generation;
        expect(validateConversationDocument(document).success).toBe(false);
    });

    it('detects aggregate overflow even when each counter is safe', () => {
        const document = emptyDocument();
        const generation = importedGeneration('generation');
        generation.usage = {
            input_tokens: Number.MAX_SAFE_INTEGER,
            output_tokens: 1,
            accounting_provenance: {
                input_tokens: provenance(),
                output_tokens: provenance(),
            },
        };
        document.generations.generation = generation;

        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'USAGE_OVERFLOW')).toBe(true);
    });

    it('accepts a consistent complete disjoint partition', () => {
        const document = emptyDocument();
        const generation = importedGeneration('generation');
        generation.usage = {
            input_tokens: 110,
            output_tokens: 5,
            total_tokens: 115,
            input_new_tokens: 25,
            cache_read_tokens: 75,
            cache_write_tokens: 10,
            input_partition: { type: 'complete_disjoint', cache_write_bucket: 'included' },
            accounting_provenance: {
                input_tokens: provenance(),
                output_tokens: provenance(),
                total_tokens: provenance('derived'),
                input_new_tokens: provenance('derived'),
                cache_read_tokens: provenance(),
                cache_write_tokens: provenance(),
            },
        };
        document.generations.generation = generation;
        expect(validateConversationDocument(document)).toMatchObject({ success: true });
    });
});

describe('builders', () => {
    it('preflights builder inputs before spreading or reading accessor properties', () => {
        let getterCalls = 0;
        const input = Object.defineProperty({ id: 'text', text: 'value', format: 'plain' as const }, 'id', {
            enumerable: true,
            get() {
                getterCalls += 1;
                return 'text';
            },
        });

        expect(() => createTextBlock(input)).toThrow();
        expect(getterCalls).toBe(0);
    });

    it('returns isolated materialized values', () => {
        const input = emptyDocument();
        const parsed = parseConversationDocument(input);
        parsed.turns.push(userTurn('later'));
        expect(input.turns).toHaveLength(0);
    });

    it('preserves exact timestamp and text values', () => {
        const block = createTextBlock({ id: 'text', text: '  punctuation: []{}\\n  ', format: 'plain' });
        expect(block.text).toBe('  punctuation: []{}\\n  ');
        expect(RECORDED_AT.endsWith('Z')).toBe(true);
    });
});
