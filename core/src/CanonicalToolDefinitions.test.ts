import type { ToolDefinition } from '@llumiverse/common';
import { deriveConversationId, fingerprintJson } from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import { canonicalToolDefinitions } from './CanonicalToolDefinitions.js';

function tool(name: string, description?: string): ToolDefinition {
    return {
        name,
        ...(description === undefined ? {} : { description }),
        input_schema: {
            type: 'object',
            properties: {
                query: { type: 'string', minLength: 1 },
                limit: { type: 'integer', minimum: 1 },
            },
            required: ['query'],
            additionalProperties: false,
        },
    };
}

describe('canonicalToolDefinitions', () => {
    it('preserves ordered schemas and exact absent-versus-empty descriptions', async () => {
        const tools = [tool('lookup'), tool('summarize', '')];
        const definitions = await canonicalToolDefinitions(tools);

        expect(definitions.map((definition) => definition.name)).toEqual(['lookup', 'summarize']);
        expect(definitions[0]).not.toHaveProperty('description');
        expect(definitions[1]).toHaveProperty('description', '');
        expect(definitions.map((definition) => definition.input_schema)).toEqual(
            tools.map((definition) => definition.input_schema),
        );
        expect(definitions[0]?.input_schema).not.toBe(tools[0]?.input_schema);

        for (const [index, definition] of definitions.entries()) {
            const source = tools[index];
            if (source === undefined) throw new Error('Missing source tool fixture');
            const version = await fingerprintJson({
                name: source.name,
                description: source.description ?? null,
                input_schema: source.input_schema,
            });
            expect(definition.version).toBe(version);
            expect(definition.id).toBe(await deriveConversationId('tool_definition', source.name, version));
        }
    });

    it('retains exact duplicates and their positions without sharing cloned schemas', async () => {
        const duplicate = tool('lookup', 'Look up a value.');
        const definitions = await canonicalToolDefinitions([duplicate, tool('middle'), duplicate]);

        expect(definitions.map((definition) => definition.name)).toEqual(['lookup', 'middle', 'lookup']);
        expect(definitions[0]).toEqual(definitions[2]);
        expect(definitions[0]).not.toBe(definitions[2]);
        expect(definitions[0]?.input_schema).not.toBe(definitions[2]?.input_schema);
    });

    it('snapshots the complete ordered catalog before asynchronous hashing', async () => {
        const first = tool('first', 'First description');
        const later = tool('later', 'Later description');
        const tools = [first, later];

        const pending = canonicalToolDefinitions(tools);

        first.name = 'mutated-first';
        first.description = 'Mutated first description';
        first.input_schema.properties = { replaced: { type: 'boolean' } };
        later.name = 'mutated-later';
        later.description = 'Mutated later description';
        later.input_schema.properties = { replaced: { type: 'number' } };
        tools.reverse();
        tools.splice(1, 0, tool('inserted'));

        const definitions = await pending;

        expect(definitions.map((definition) => definition.name)).toEqual(['first', 'later']);
        expect(definitions.map((definition) => definition.description)).toEqual([
            'First description',
            'Later description',
        ]);
        expect(definitions.map((definition) => definition.input_schema)).toEqual([
            tool('first').input_schema,
            tool('later').input_schema,
        ]);
        for (const definition of definitions) {
            expect(definition.version).toBe(
                await fingerprintJson({
                    name: definition.name,
                    description: definition.description ?? null,
                    input_schema: definition.input_schema,
                }),
            );
            expect(definition.id).toBe(
                await deriveConversationId('tool_definition', definition.name, definition.version),
            );
        }
        expect(tools.map((definition) => definition.name)).toEqual(['mutated-later', 'inserted', 'mutated-first']);
        expect(first.input_schema.properties).toEqual({ replaced: { type: 'boolean' } });
        expect(later.input_schema.properties).toEqual({ replaced: { type: 'number' } });
    });

    it('preserves boolean schemas', async () => {
        const booleanSchemaTool = { name: 'disabled', input_schema: false } as unknown as ToolDefinition;

        const [definition] = await canonicalToolDefinitions([booleanSchemaTool]);

        expect(definition?.input_schema).toBe(false);
        expect(definition?.version).toBe(
            await fingerprintJson({ name: 'disabled', description: null, input_schema: false }),
        );
    });

    it('rejects a non-JSON schema value during the existing fingerprint preflight', async () => {
        const invalid = tool('invalid');
        invalid.input_schema.properties = { query: { type: 'string', default: undefined } };

        await expect(canonicalToolDefinitions([invalid])).rejects.toThrow('Fingerprint input failed JSON preflight');
    });
});
