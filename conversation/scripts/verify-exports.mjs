import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFile, rm } from 'node:fs/promises';

const root = await import('@llumiverse/conversation');
const schemas = await import('@llumiverse/conversation/schemas');
const jsonSchemas = await import('@llumiverse/conversation/json-schema');

assert.equal(typeof root.validateConversationDocument, 'function');
assert.equal(typeof root.createConversationDocument, 'function');
assert.equal(typeof schemas.ConversationDocumentSchema?.safeParse, 'function');
assert.equal(jsonSchemas.ConversationDocumentJsonSchema.$schema, 'https://json-schema.org/draft/2020-12/schema');

const compiler = spawnSync('pnpm', ['exec', 'tsc', '-p', 'test-fixtures/tsconfig.json'], {
    cwd: new URL('..', import.meta.url),
    encoding: 'utf8',
});
assert.equal(compiler.status, 0, compiler.stderr || compiler.stdout);

const outputUrl = new URL('../.verify/type-only.js', import.meta.url);
const output = await readFile(outputUrl, 'utf8');
assert.doesNotMatch(output, /@llumiverse\/conversation|zod/);
await rm(new URL('../.verify', import.meta.url), { recursive: true, force: true });
