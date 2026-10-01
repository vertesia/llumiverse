import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFile, rm } from 'node:fs/promises';

const root = await import('@llumiverse/conversation');
const schemas = await import('@llumiverse/conversation/schemas');
const jsonSchemas = await import('@llumiverse/conversation/json-schema');
const outputRuntime = await import('@llumiverse/conversation/output-runtime');
const streamingRuntime = await import('@llumiverse/conversation/streaming-runtime');

assert.equal(typeof root.validateConversationDocument, 'function');
assert.equal(typeof root.createConversationDocument, 'function');
assert.equal(typeof schemas.ConversationDocumentSchema?.safeParse, 'function');
assert.equal(typeof schemas.ConversationStreamEventSchema?.safeParse, 'function');
assert.equal(typeof schemas.ConversationStreamCursorSchema?.safeParse, 'function');
assert.equal(typeof schemas.ConversationStreamIdentitySchema?.safeParse, 'function');
assert.equal(typeof schemas.ConversationTranscriptFragmentSchema?.safeParse, 'function');
assert.equal(typeof schemas.ConversationTranscriptGenerationSchema?.safeParse, 'function');
assert.equal(typeof root.createConversationTranscriptFragment, 'function');
assert.equal(jsonSchemas.ConversationDocumentJsonSchema.$schema, 'https://json-schema.org/draft/2020-12/schema');
assert.equal(jsonSchemas.ConversationStreamEventJsonSchema.$schema, 'https://json-schema.org/draft/2020-12/schema');
assert.equal(jsonSchemas.ConversationStreamCursorJsonSchema.$schema, 'https://json-schema.org/draft/2020-12/schema');
assert.equal(jsonSchemas.ConversationStreamIdentityJsonSchema.$schema, 'https://json-schema.org/draft/2020-12/schema');
assert.equal(
    jsonSchemas.ConversationTranscriptFragmentJsonSchema.$schema,
    'https://json-schema.org/draft/2020-12/schema',
);
assert.equal(typeof outputRuntime.cloneSemanticallyValidAcceptedOutputFragment, 'function');
assert.equal(typeof outputRuntime.conversationOutputReceiptsEqual, 'function');
assert.equal(typeof streamingRuntime.ConversationStreamAccumulatorRuntime, 'function');
assert.equal(typeof streamingRuntime.preflightJsonInput, 'function');

for (const name of ['Options', 'Diagnostic', 'Report', 'Result']) {
    assert.equal(typeof schemas[`NativeConversationImport${name}Schema`]?.safeParse, 'function');
    assert.equal(
        jsonSchemas[`NativeConversationImport${name}JsonSchema`].$schema,
        'https://json-schema.org/draft/2020-12/schema',
    );
}
assert.equal(typeof root.parseNativeConversationImportOptions, 'function');
assert.equal(typeof root.parseNativeConversationImportReport, 'function');
assert.equal(typeof root.parseNativeConversationImportResult, 'function');
assert.equal(typeof root.preflightNativeConversationImportInput, 'function');

const compiler = spawnSync('pnpm', ['exec', 'tsc', '-p', 'test-fixtures/tsconfig.json'], {
    cwd: new URL('..', import.meta.url),
    encoding: 'utf8',
});
assert.equal(compiler.status, 0, compiler.stderr || compiler.stdout);

const outputUrl = new URL('../.verify/type-only.js', import.meta.url);
const output = await readFile(outputUrl, 'utf8');
assert.doesNotMatch(output, /@llumiverse\/conversation|zod/);
await rm(new URL('../.verify', import.meta.url), { recursive: true, force: true });
