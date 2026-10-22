import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { verifyPackageTarball } from './verify-package-tarball.mjs';

function fixture(t, overrides = {}, files = ['lib/index.js', 'lib/index.d.ts']) {
    const directory = mkdtempSync(join(tmpdir(), 'package-verifier-'));
    t.after(() => rmSync(directory, { recursive: true, force: true }));
    mkdirSync(join(directory, 'package'));
    writeFileSync(
        join(directory, 'package/package.json'),
        JSON.stringify({
            name: '@llumiverse/example',
            version: '1.6.0',
            main: './lib/index.js',
            types: './lib/index.d.ts',
            exports: { '.': { types: './lib/index.d.ts', default: './lib/index.js' } },
            ...overrides,
        }),
    );
    for (const file of files) {
        mkdirSync(dirname(join(directory, 'package', file)), { recursive: true });
        writeFileSync(join(directory, 'package', file), 'fixture');
    }
    const tarball = join(directory, 'package.tgz');
    execFileSync('tar', ['-czf', tarball, '-C', directory, 'package']);
    return tarball;
}

test('flat lib output passes without obsolete esm/cjs directories', (t) => {
    assert.equal(verifyPackageTarball(fixture(t), '1.6.0'), '@llumiverse/example');
});
test('missing runtime and type entry points fail', (t) => {
    for (const missing of ['lib/index.js', 'lib/index.d.ts']) {
        const files = ['lib/index.js', 'lib/index.d.ts'].filter((file) => file !== missing);
        assert.throws(() => verifyPackageTarball(fixture(t, {}, files), '1.6.0'), /missing from the tarball/);
    }
});
test('every conditional export and optional module target is checked', (t) => {
    for (const overrides of [
        { module: './lib/missing.js' },
        { exports: { '.': { import: './lib/index.js', require: './lib/missing.cjs' } } },
    ]) {
        assert.throws(() => verifyPackageTarball(fixture(t, overrides), '1.6.0'), /missing from the tarball/);
    }
});
test('wildcard exports match packed files and null exports are ignored', (t) => {
    assert.equal(
        verifyPackageTarball(fixture(t, { exports: { './*': './lib/*.js', './private': null } }), '1.6.0'),
        '@llumiverse/example',
    );
    assert.throws(
        () => verifyPackageTarball(fixture(t, { exports: './missing/*.js' }), '1.6.0'),
        /missing from the tarball/,
    );
});
test('root version and every cohort dependency must match', (t) => {
    assert.throws(() => verifyPackageTarball(fixture(t, { version: '1.5.0' }), '1.6.0'), /expected version/);
    assert.throws(
        () =>
            verifyPackageTarball(
                fixture(t, { dependencies: { '@llumiverse/common': '1.6.0', '@llumiverse/core': '1.5.0' } }),
                '1.6.0',
            ),
        /@llumiverse\/core must use cohort version/,
    );
});
test('targets outside the package are rejected', (t) => {
    assert.throws(() => verifyPackageTarball(fixture(t, { main: '../outside.js' }), '1.6.0'), /Invalid package target/);
});
