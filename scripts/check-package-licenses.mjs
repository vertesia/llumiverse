#!/usr/bin/env node
// Checks the license of every package in this repository: each package.json declares Apache-2.0 (or the
// license listed below), and each published package carries a LICENSE file with the full text of that
// license. pnpm copies the repository's Apache-2.0 LICENSE into a package that has none, so a package
// under another license must carry its own.
//
//   node scripts/check-package-licenses.mjs

import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_LICENSE = 'Apache-2.0';

// Packages under another license. Add one only with the reason.
const OTHER_LICENSES = new Map();

// Packages whose LICENSE file does not yet match their package.json. Each needs a decision; remove the
// entry once the two agree.
const KNOWN_MISMATCHES = new Map();

// Folders that are not packages of this repository.
const NOT_CHECKED = new Map();

const FULL_TEXT = {
    'Apache-2.0': (text) =>
        /Apache License\s+Version 2\.0, January 2004/.test(text) && /END OF TERMS AND CONDITIONS/.test(text),
    MIT: (text) => /Permission is hereby granted, free of charge/.test(text),
};

function manifests() {
    return execFileSync('git', ['ls-files', '*package.json'], { cwd: ROOT, encoding: 'utf8' })
        .split('\n')
        .filter(Boolean)
        .filter(
            (path) =>
                ![...NOT_CHECKED.keys()].some((dir) => path === `${dir}/package.json` || path.startsWith(`${dir}/`)),
        );
}

function licenseText(dir) {
    const file = readdirSync(join(ROOT, dir)).find((name) => /^licen[cs]e(\.(md|txt))?$/i.test(name));
    return file ? readFileSync(join(ROOT, dir, file), 'utf8') : undefined;
}

const problems = [];
for (const path of manifests()) {
    const dir = dirname(path);
    const pkg = JSON.parse(readFileSync(join(ROOT, path), 'utf8'));
    const expected = OTHER_LICENSES.get(pkg.name)?.license ?? DEFAULT_LICENSE;
    if (pkg.license !== expected) {
        problems.push(`${path}: "license" is ${JSON.stringify(pkg.license)}, expected "${expected}"`);
        continue;
    }
    if (pkg.private === true || KNOWN_MISMATCHES.has(pkg.name)) {
        continue;
    }
    const text = licenseText(dir);
    if (text === undefined) {
        const from = expected === DEFAULT_LICENSE ? ' from the repository root' : '';
        problems.push(`${dir}: no LICENSE file; add the full ${expected} text${from}`);
    } else if (!FULL_TEXT[expected](text)) {
        problems.push(`${dir}/LICENSE: not the full ${expected} text`);
    }
}
for (const name of KNOWN_MISMATCHES.keys()) {
    if (!manifests().some((path) => JSON.parse(readFileSync(join(ROOT, path), 'utf8')).name === name)) {
        problems.push(`KNOWN_MISMATCHES lists ${name}, which no package is named`);
    }
}

if (problems.length > 0) {
    console.error(`Package licenses that do not match the repository's (${DEFAULT_LICENSE}):`);
    for (const problem of problems) {
        console.error(`  ${problem}`);
    }
    process.exit(1);
}
console.log(`Every package declares its license and ships its full text.`);
