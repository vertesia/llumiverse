import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { verifyPackedPackage, verifyPublicationPlan } from './verify-package-release.mjs';

const version = '1.6.0-dev.20260930.000000Z';
const conversation = {
    name: '@llumiverse/conversation',
    version,
    type: 'module',
    main: './lib/index.js',
    types: './lib/index.d.ts',
    exports: { '.': { types: './lib/index.d.ts', default: './lib/index.js' }, './schemas': './lib/schemas/index.js' },
};
const common = { name: '@llumiverse/common', version, dependencies: { '@llumiverse/conversation': 'workspace:*' } };
const files = ['package/lib/index.js', 'package/lib/index.d.ts', 'package/lib/schemas/index.js'];

test('orders the conversation package before every consuming public package', () => {
    verifyPublicationPlan([conversation, common]);
    assert.throws(() => verifyPublicationPlan([common, conversation]), /must be published before/);
    assert.throws(() => verifyPublicationPlan([common]), /unpublished workspace package/);
});

test('blocks the incomplete private conversation package before publication', () => {
    assert.throws(() => verifyPublicationPlan([{ ...conversation, private: true }, common]), /is private/);
});

test('verifies actual ESM entry points without requiring obsolete CJS folders', () => {
    verifyPackedPackage(conversation, files, conversation.name, version);
    assert.throws(() => verifyPackedPackage(conversation, files.slice(0, 2), conversation.name, version), /schemas/);
});

test('rejects each mismatched internal dependency even when another dependency matches', () => {
    const pkg = {
        ...conversation,
        dependencies: { '@llumiverse/common': version, '@llumiverse/core': '1.0.0' },
    };
    assert.throws(() => verifyPackedPackage(pkg, files, pkg.name, version), /mismatched dependency @llumiverse\/core/);
});

test('rejects unresolved workspace and catalog dependencies in tarballs', () => {
    for (const range of ['workspace:*', 'catalog:']) {
        assert.throws(
            () =>
                verifyPackedPackage(
                    { ...conversation, dependencies: { zod: range } },
                    files,
                    conversation.name,
                    version,
                ),
            /unresolved dependency/,
        );
    }
});

test('rejects a wrong package, version, private flag or absent exports', () => {
    for (const changed of [
        { name: '@llumiverse/other' },
        { version: '1.0.0' },
        { private: true },
        { exports: undefined },
    ]) {
        assert.throws(() => verifyPackedPackage({ ...conversation, ...changed }, files, conversation.name, version));
    }
});

test('publishing an incomplete workspace stops before version or publish commands', () => {
    const fixture = mkdtempSync(join(tmpdir(), 'llumiverse-publish-'));
    try {
        const bin = join(fixture, '.github/bin');
        mkdirSync(bin, { recursive: true });
        for (const filename of ['publish-all-packages.sh', 'verify-package-release.mjs', 'lib-package-channel.sh']) {
            writeFileSync(join(bin, filename), readFileSync(new URL(filename, import.meta.url)));
        }
        for (const name of ['conversation', 'common', 'core', 'drivers']) {
            mkdirSync(join(fixture, name));
            writeFileSync(
                join(fixture, name, 'package.json'),
                JSON.stringify({
                    name: `@llumiverse/${name}`,
                    version,
                    ...(name === 'conversation' ? { private: true } : {}),
                }),
            );
        }
        assert.throws(
            () =>
                execFileSync(
                    'bash',
                    [
                        join(bin, 'publish-all-packages.sh'),
                        '--ref',
                        'main',
                        '--release-type',
                        'snapshot',
                        '--bump-type',
                        'keep',
                        '--dry-run',
                    ],
                    {
                        cwd: fixture,
                        encoding: 'utf8',
                        stdio: 'pipe',
                    },
                ),
            (error) => {
                assert.equal(error.status, 1);
                assert.match(error.stderr, /@llumiverse\/conversation is private/);
                assert.doesNotMatch(error.stdout, /Updating package versions|Building all packages|Publishing @/);
                return true;
            },
        );
        assert.equal(JSON.parse(readFileSync(join(fixture, 'conversation/package.json'), 'utf8')).version, version);
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});
