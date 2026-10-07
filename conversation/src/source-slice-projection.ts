import { ConversationValidationError } from './diagnostics.js';
import { pointerFromTokens, pointerPrefix, pointerTokens, resolveJsonPointer } from './json-pointer.js';
import { preflightJsonInput } from './json-preflight.js';
import { JsonValueSchema } from './schemas/primitives.js';
import { JsonSourceRegionSchema } from './schemas/source-slices.js';
import { rejectSourceSlice, type SourceSliceWork, spendSourceSliceWork } from './source-slice-work.js';
import type { JsonInverseMapping, JsonSourceRegion, JsonValue } from './types.js';

/** Exact JSON subtree projection; exclusion is absence, never a fabricated null placeholder. */
export function projectJsonSourceRegion(
    value: JsonValue,
    region: JsonSourceRegion,
): { value: JsonValue; inverse: JsonInverseMapping } {
    const preflight = preflightJsonInput({ value, region });
    if (!preflight.success)
        throw new ConversationValidationError('JSON projection input failed preflight', preflight.diagnostics);
    const ownedValue = JsonValueSchema.parse(value);
    const ownedRegion = JsonSourceRegionSchema.parse(region);
    return projectOwnedJsonSourceRegion(ownedValue, ownedRegion, { nodes: 0 });
}
/** Internal: inputs already belong to a preflighted snapshot. One budget spans all lineage groups. */
export function projectOwnedJsonSourceRegion(
    value: JsonValue,
    region: JsonSourceRegion,
    work: SourceSliceWork,
): { value: JsonValue; inverse: JsonInverseMapping } {
    const objectOrder = new WeakMap<object, ReadonlyMap<string, number>>();
    spendSourceSliceWork(work, pointerTokens(region.pointer).length + 1);
    const root = resolveJsonPointer(value, region.pointer, objectOrder);
    const exclusions = region.excluded_pointers
        .map((pointer) => {
            spendSourceSliceWork(work, pointerTokens(pointer).length + 1);
            const resolved = resolveJsonPointer(value, pointer, objectOrder);
            if (!pointerPrefix(root.segments, resolved.segments) || root.segments.length === resolved.segments.length)
                throw new Error('JSON exclusion must be a strict source-region descendant');
            return resolved;
        })
        .sort((a, b) => {
            for (let index = 0; index < Math.min(a.order.length, b.order.length); index++)
                if (a.order[index] !== b.order[index]) return a.order[index] - b.order[index];
            return a.order.length - b.order.length;
        });
    for (let index = 0; index < exclusions.length; index++)
        for (let prior = 0; prior < index; prior++) {
            spendSourceSliceWork(work);
            if (pointerPrefix(exclusions[prior].segments, exclusions[index].segments))
                rejectSourceSlice('JSON exclusions overlap or repeat');
        }
    if (exclusions.some((item, index) => pointerFromTokens(item.segments) !== region.excluded_pointers[index]))
        throw new Error('JSON exclusions must follow original source order');
    const excluded = new Set(region.excluded_pointers),
        inverse: JsonInverseMapping = { root_source_pointer: region.pointer, arrays: [] };
    const visit = (node: JsonValue, sourceTokens: string[], targetTokens: string[]): JsonValue => {
        spendSourceSliceWork(work);
        if (node === null || typeof node !== 'object') return node;
        if (Array.isArray(node)) {
            const result: JsonValue[] = [],
                runs: JsonInverseMapping['arrays'][number]['runs'] = [];
            for (let index = 0; index < node.length; index++) {
                const source = [...sourceTokens, String(index)];
                if (excluded.has(pointerFromTokens(source))) continue;
                const targetIndex = result.length,
                    last = runs.at(-1);
                if (
                    last &&
                    last.source_start + last.length === index &&
                    last.target_start + last.length === targetIndex
                )
                    last.length++;
                else runs.push({ source_start: index, target_start: targetIndex, length: 1 });
                result.push(visit(node[index], source, [...targetTokens, String(targetIndex)]));
            }
            if (inverse.arrays.length >= 4096 || runs.length > 4096)
                rejectSourceSlice('JSON inverse mapping exceeds bounded record limits');
            inverse.arrays.push({
                target_pointer: pointerFromTokens(targetTokens),
                source_pointer: pointerFromTokens(sourceTokens),
                runs,
            });
            return result;
        }
        const entries: [string, JsonValue][] = [];
        for (const key of Object.keys(node).sort()) {
            const source = [...sourceTokens, key];
            if (!excluded.has(pointerFromTokens(source)))
                entries.push([key, visit(node[key], source, [...targetTokens, key])]);
        }
        return Object.fromEntries(entries);
    };
    return { value: visit(root.value, root.segments, []), inverse };
}
export { inverseJsonPointer } from './json-pointer-inverse.js';
