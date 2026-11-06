import { execFileSync } from 'node:child_process';
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const tsc = join(dirname(createRequire(import.meta.url).resolve('typescript/package.json')), 'bin', 'tsc');
const schemaModule = fileURLToPath(new URL('./model-options.js', import.meta.url));

/** Emits the declaration a downstream package would publish for `source`, which imports this module. */
function emitDeclaration(source: string): string {
    const dir = mkdtempSync(join(tmpdir(), 'model-options-dts-'));
    try {
        const consumer = join(dir, 'consumer.ts');
        writeFileSync(consumer, source);
        execFileSync(process.execPath, [
            tsc,
            '--ignoreConfig',
            '--declaration',
            '--emitDeclarationOnly',
            '--strict',
            '--skipLibCheck',
            '--module',
            'nodenext',
            '--moduleResolution',
            'nodenext',
            '--target',
            'es2022',
            '--outDir',
            join(dir, 'out'),
            consumer,
        ]);
        const emitted = readdirSync(join(dir, 'out'), { recursive: true, encoding: 'utf8' }).find((file) =>
            file.endsWith('consumer.d.ts'),
        );
        if (!emitted) throw new Error('tsc emitted no declaration for the consumer');
        return readFileSync(join(dir, 'out', emitted), 'utf8');
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}

/**
 * The type checker cannot see this property: an interface and the inferred union it extends are the same
 * type to it. Only declaration emit differs, by printing the inferred union in full at every use, so the
 * test compiles a consumer and reads the declaration it publishes.
 */
describe('ModelOptionsSchema declarations', () => {
    it('reference the union by name where another schema embeds it', { timeout: 60_000 }, () => {
        const declaration = emitDeclaration(
            `import { ModelOptionsSchema } from ${JSON.stringify(schemaModule)};\n` +
                'export const Embedded = { direct: ModelOptionsSchema, optional: ModelOptionsSchema.optional() };\n',
        );

        const unnamed = 'the embedded union is printed in full: keep the `ModelOptionsSchemaType` annotation';
        expect(declaration, unnamed).toContain('ModelOptionsSchemaType');
        // Printed in full, the union spells out every member, including this family ID.
        expect(declaration, unnamed).not.toContain('openai-thinking');
    });
});
