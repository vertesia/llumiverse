import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { test } from 'node:test';

// CI installs the Node version in .node-version. Workflows check out other refs too (a release-line
// PR head, a release branch to update), and older release refs predate the file, so every workflow
// setup-node step pairs node-version-file with a fallback to the same major for when the file is
// absent. A step that drops back to a literal version would quietly stop following .node-version, so
// these tests hold every step to the exact pair.

const root = new URL('../../', import.meta.url);
const read = (path) => readFileSync(new URL(path, root), 'utf8');

const nodeVersion = read('.node-version');
const major = nodeVersion.split('.')[0];
const fallbackPair = {
    'node-version': `\${{ hashFiles('.node-version') == '' && '${major}' || '' }}`,
    'node-version-file': "${{ hashFiles('.node-version') != '' && '.node-version' || '' }}",
};

// The `with:` inputs of every actions/setup-node step, read from the YAML text: steps follow one
// layout, a `- ` list item whose keys share the column of `uses:`.
function setupNodeSteps(text) {
    const lines = text.split('\n');
    const indent = (line) => line.length - line.trimStart().length;
    const steps = [];
    lines.forEach((line, index) => {
        if (!/^\s*(- )?uses:\s*actions\/setup-node@/.test(line)) {
            return;
        }
        const keyCol = line.indexOf('uses:');
        let start = index;
        while (start > 0 && !lines[start].startsWith(`${' '.repeat(keyCol - 2)}- `)) {
            start--;
        }
        let end = index + 1;
        while (end < lines.length && (lines[end].trim() === '' || indent(lines[end]) >= keyCol)) {
            end++;
        }
        const block = lines.slice(start, end);
        const withAt = block.findIndex((l) => l.slice(keyCol).startsWith('with:'));
        const inputs = {};
        if (withAt !== -1) {
            for (const l of block.slice(withAt + 1)) {
                const trimmed = l.trim();
                if (trimmed === '' || trimmed.startsWith('#')) {
                    continue;
                }
                if (indent(l) <= keyCol) {
                    break;
                }
                const match = trimmed.match(/^([A-Za-z0-9_-]+):\s*(.*?)\s*$/);
                if (match) {
                    inputs[match[1]] = match[2];
                }
            }
        }
        steps.push({ line: index + 1, inputs });
    });
    return steps;
}

test('.node-version pins an exact version', () => {
    assert.match(nodeVersion, /^\d+\.\d+\.\d+\n$/);
});

test('every workflow setup-node step pairs node-version-file with a fallback to the same major', () => {
    const workflows = readdirSync(new URL('.github/workflows/', root)).filter((name) => /\.ya?ml$/.test(name));
    let checked = 0;
    for (const name of workflows) {
        const file = `.github/workflows/${name}`;
        const text = read(file);
        const steps = setupNodeSteps(text);
        // Every reference has to be a step the reader understood; a quoted or reformatted `uses:` that
        // it skipped would otherwise escape the check.
        assert.equal(steps.length, text.match(/actions\/setup-node@/g)?.length ?? 0, file);
        for (const { line, inputs } of steps) {
            assert.deepEqual(
                { 'node-version': inputs['node-version'], 'node-version-file': inputs['node-version-file'] },
                fallbackPair,
                `${file}:${line}`,
            );
            checked++;
        }
    }
    assert.ok(checked > 0, 'no setup-node steps found');
});

test('the step reader keeps each step to its own inputs', () => {
    const workflow = [
        'jobs:',
        '  build:',
        '    steps:',
        '      - uses: actions/checkout@abc # v7',
        '      - name: Install Node.js',
        '        uses: actions/setup-node@def # v7',
        '        with:',
        '          # A comment between inputs',
        '          node-version: 24',
        '          cache: pnpm',
        '      - uses: actions/setup-node@def # v7',
        '        with:',
        "          node-version-file: ${{ hashFiles('.node-version') != '' && '.node-version' || '' }}",
        '      - name: Next step',
        '        run: echo',
    ].join('\n');
    const steps = setupNodeSteps(workflow);
    assert.deepEqual(steps, [
        { line: 6, inputs: { 'node-version': '24', cache: 'pnpm' } },
        { line: 11, inputs: { 'node-version-file': fallbackPair['node-version-file'] } },
    ]);
    // A literal version is what the pair check exists to reject.
    assert.notDeepEqual(steps[0].inputs, fallbackPair);
});
