import { ConversationValidationError } from './diagnostics.js';
import { JSON_POINTER_PATTERN_SOURCE } from './runtime-constants.js';
import type { JsonValue } from './types.js';

const jsonPointerPattern = new RegExp(JSON_POINTER_PATTERN_SOURCE);

export function pointerTokens(pointer: string): string[] {
    if (typeof pointer !== 'string' || !jsonPointerPattern.test(pointer))
        throw new TypeError('Invalid RFC 6901 JSON Pointer');
    return pointer === ''
        ? []
        : pointer
              .slice(1)
              .split('/')
              .map((token) => token.replace(/~1/g, '/').replace(/~0/g, '~'));
}
export function pointerFromTokens(tokens: readonly string[]): string {
    return tokens.length ? `/${tokens.map((token) => token.replace(/~/g, '~0').replace(/\//g, '~1')).join('/')}` : '';
}
export function pointerPrefix(first: readonly string[], second: readonly string[]): boolean {
    return first.length <= second.length && first.every((token, index) => token === second[index]);
}
export function resolveJsonPointer(
    value: JsonValue,
    pointer: string,
    objectOrder = new WeakMap<object, ReadonlyMap<string, number>>(),
) {
    const segments = pointerTokens(pointer),
        order: number[] = [];
    let current = value;
    function reject(message: string): never {
        throw new ConversationValidationError(message, [
            { code: 'REFERENCE_NOT_FOUND', stage: 'semantic', path: '/', message },
        ]);
    }
    for (const token of segments) {
        if (current === null || typeof current !== 'object' || !Object.hasOwn(current, token))
            reject('JSON Pointer target is unavailable');
        if (Array.isArray(current)) {
            if (!/^(?:0|[1-9]\d*)$/.test(token)) reject('JSON Pointer array index must be canonical');
            const index = Number(token);
            if (!Number.isSafeInteger(index) || index >= current.length)
                reject('JSON Pointer array index is out of bounds');
            order.push(index);
            current = current[index];
        } else {
            let positions = objectOrder.get(current);
            if (!positions) {
                positions = new Map(
                    Object.keys(current)
                        .sort()
                        .map((key, index) => [key, index]),
                );
                objectOrder.set(current, positions);
            }
            const position = positions.get(token);
            if (position === undefined) reject('JSON Pointer property is unavailable');
            order.push(position);
            current = current[token];
        }
    }
    return { segments, order, value: current };
}
