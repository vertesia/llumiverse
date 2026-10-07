import { pointerFromTokens, pointerTokens } from './json-pointer.js';
import { type SourceSliceWork, spendSourceSliceWork } from './source-slice-work.js';
import type { JsonInverseMapping } from './types.js';

/** Canonical target pointer maps to its original source position, even after nested array shifts. */
export function inverseJsonPointer(
    mapping: JsonInverseMapping,
    pointer: string,
    work: SourceSliceWork = { nodes: 0 },
): string {
    spendSourceSliceWork(work, mapping.arrays.length + pointerTokens(pointer).length);
    const target = pointerTokens(pointer),
        root = pointerTokens(mapping.root_source_pointer);
    const arrays = new Map(mapping.arrays.map((entry) => [entry.target_pointer, entry]));
    let source = [...root];
    for (let index = 0; index < target.length; index++) {
        const array = arrays.get(pointerFromTokens(target.slice(0, index)));
        if (array) {
            if (!/^(?:0|[1-9]\d*)$/.test(target[index])) throw new Error('Inverse JSON array index must be canonical');
            const position = Number(target[index]),
                run = array.runs.find(
                    (run) => position >= run.target_start && position < run.target_start + run.length,
                );
            if (!run) throw new Error('Inverse JSON pointer is unavailable');
            source = [...pointerTokens(array.source_pointer), String(run.source_start + position - run.target_start)];
        } else source.push(target[index]);
    }
    return pointerFromTokens(source);
}
