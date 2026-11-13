import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import * as schemas from './index.js';
import { buildLlumiverseJsonSchema, LLUMIVERSE_JSON_SCHEMA_PATH } from './json-schema-document.js';

const committedPath = fileURLToPath(new URL(`../../${LLUMIVERSE_JSON_SCHEMA_PATH}`, import.meta.url));

describe('llumiverse JSON Schema document', () => {
    const document = buildLlumiverseJsonSchema();
    const defs = document.$defs as Record<string, Record<string, unknown>>;

    it('matches the committed copy (run `pnpm gen:json-schema` after editing a schema)', async () => {
        await expect(`${JSON.stringify(document, null, 4)}\n`).toMatchFileSnapshot(committedPath);
    });

    it('defines every exported schema', () => {
        const exported = Object.entries(schemas).filter(
            ([name, value]) => name.endsWith('Schema') && value instanceof z.ZodType,
        );
        expect(Object.keys(defs)).toHaveLength(exported.length);
    });

    it('keeps the names the Vertesia API contract publishes', () => {
        for (const name of ['PromptSegment', 'ToolUse', 'CompletionResult', 'ExecutionTokenUsage', 'ModelOptions']) {
            expect(defs).toHaveProperty(name);
        }
    });

    it('references shared shapes instead of inlining them', () => {
        const properties = defs.PromptSegment.properties as Record<string, Record<string, unknown>>;
        expect(properties.role).toEqual({ $ref: '#/$defs/PromptRole' });
    });

    it('leaves no reference unresolved', () => {
        const refs = JSON.stringify(document).match(/"\$ref":"#\/\$defs\/[^"]+"/g) ?? [];
        for (const ref of refs) {
            expect(defs).toHaveProperty(ref.slice('"$ref":"#/$defs/'.length, -1));
        }
    });
});
