import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

export function verifyPackageTarball(tarball, expectedVersion) {
    const files = new Set(
        execFileSync('tar', ['-tzf', tarball], { encoding: 'utf8' })
            .trim()
            .split('\n')
            .filter((name) => !name.endsWith('/')),
    );
    const manifest = JSON.parse(execFileSync('tar', ['-xzOf', tarball, 'package/package.json'], { encoding: 'utf8' }));
    if (manifest.version !== expectedVersion) {
        throw new Error(`${manifest.name}: expected version ${expectedVersion}, got ${manifest.version}`);
    }
    for (const section of ['dependencies', 'optionalDependencies']) {
        for (const [name, version] of Object.entries(manifest[section] ?? {})) {
            if (name.startsWith('@llumiverse/') && version !== expectedVersion) {
                throw new Error(`${manifest.name}: ${name} must use cohort version ${expectedVersion}, got ${version}`);
            }
        }
    }
    const targets = [];
    function collect(value) {
        if (typeof value === 'string') targets.push(value);
        else if (value && typeof value === 'object') Object.values(value).forEach(collect);
    }
    for (const value of [manifest.main, manifest.module, manifest.types, manifest.typings, manifest.exports])
        collect(value);
    if (!targets.length) throw new Error(`${manifest.name}: no declared package entry points`);
    for (const target of targets) {
        const name = target.replace(/^\.\//, '');
        if (name.startsWith('/') || name.split('/').includes('..'))
            throw new Error(`Invalid package target: ${target}`);
        const pattern = new RegExp(
            `^package/${name
                .split('*')
                .map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
                .join('.*')}$`,
        );
        if (![...files].some((file) => pattern.test(file))) {
            throw new Error(`${manifest.name}: declared target ${target} is missing from the tarball`);
        }
    }
    return manifest.name;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
    try {
        console.log(
            `Verified ${verifyPackageTarball(process.argv[2], process.argv[3])}: version, dependencies and entry points`,
        );
    } catch (error) {
        console.error(error.message);
        process.exitCode = 1;
    }
}
