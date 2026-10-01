import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** Validate the complete publication set before versions, Git state or npm are changed. */
export function verifyPublicationPlan(packages) {
    const byName = new Map(packages.map((pkg) => [pkg.name, pkg]));
    if (byName.size !== packages.length) throw new Error('Duplicate package in publication plan');
    const available = new Set();
    for (const pkg of packages) {
        if (typeof pkg.name !== 'string' || !pkg.name.startsWith('@llumiverse/')) {
            throw new Error('Publication plan contains an invalid package name');
        }
        if (pkg.private) throw new Error(`${pkg.name} is private; its adoption and release gate is not complete`);
        for (const dependencies of [pkg.dependencies, pkg.optionalDependencies, pkg.peerDependencies]) {
            for (const [name, version] of Object.entries(dependencies ?? {})) {
                if (!version.startsWith('workspace:')) continue;
                if (!byName.has(name)) throw new Error(`${pkg.name} depends on unpublished workspace package ${name}`);
                if (!available.has(name)) throw new Error(`${name} must be published before ${pkg.name}`);
            }
        }
        available.add(pkg.name);
    }
}

function exportTargets(value) {
    if (typeof value === 'string') return [value];
    if (value === null || value === undefined) return [];
    return Object.values(value).flatMap(exportTargets);
}

/** Check packed metadata and every declared entry point, rather than obsolete output directories. */
export function verifyPackedPackage(pkg, files, expectedName, expectedVersion) {
    if (pkg.name !== expectedName) throw new Error(`Expected ${expectedName}, found ${pkg.name}`);
    if (pkg.version !== expectedVersion) throw new Error(`${pkg.name} has unexpected version ${pkg.version}`);
    if (pkg.private) throw new Error(`${pkg.name} is private`);
    for (const dependencies of [pkg.dependencies, pkg.optionalDependencies, pkg.peerDependencies]) {
        for (const [name, version] of Object.entries(dependencies ?? {})) {
            if (version.startsWith('workspace:') || version.startsWith('catalog:')) {
                throw new Error(`${pkg.name} has unresolved dependency ${name}: ${version}`);
            }
            if (name.startsWith('@llumiverse/') && version !== expectedVersion) {
                throw new Error(`${pkg.name} has mismatched dependency ${name}: ${version}`);
            }
        }
    }
    if (!pkg.main || !pkg.types || !pkg.exports) throw new Error(`${pkg.name} is missing declared entry points`);
    const targets = new Set([pkg.main, pkg.types, ...exportTargets(pkg.exports)]);
    const packedFiles = new Set(files);
    for (const target of targets) {
        if (!target.startsWith('./') || target.includes('..') || target.includes('*')) {
            throw new Error(`${pkg.name} has an unsupported entry point ${target}`);
        }
        if (!packedFiles.has(`package/${target.slice(2)}`)) {
            throw new Error(`${pkg.name} is missing packed entry point ${target}`);
        }
    }
}

function main(args) {
    const [command, ...rest] = args;
    if (command === 'preflight') {
        const [root, ...directories] = rest;
        if (!root || directories.length === 0) throw new Error('preflight requires a root and ordered packages');
        verifyPublicationPlan(
            directories.map((directory) => JSON.parse(readFileSync(resolve(root, directory, 'package.json'), 'utf8'))),
        );
    } else if (command === 'tarball') {
        const [path, name, version] = rest;
        if (!path || !name || !version) throw new Error('tarball requires a path, package name and version');
        const pkg = JSON.parse(execFileSync('tar', ['-xzOf', path, 'package/package.json'], { encoding: 'utf8' }));
        const files = execFileSync('tar', ['-tzf', path], { encoding: 'utf8' }).trim().split('\n');
        verifyPackedPackage(pkg, files, name, version);
    } else {
        throw new Error('Expected preflight or tarball command');
    }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    try {
        main(process.argv.slice(2));
    } catch (error) {
        console.error(error instanceof Error ? error.message : String(error));
        process.exitCode = 1;
    }
}
