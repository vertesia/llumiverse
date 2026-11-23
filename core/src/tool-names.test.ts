import { describe, expect, it } from 'vitest';
import { declaredToolNames, providerSafeToolName } from './tool-names.js';

describe('providerSafeToolName', () => {
    const declared = declaredToolNames([{ name: 'web_search' }, { name: 'mcp.github.search' }]);

    it('keeps a declared name verbatim, even one with rejected characters', () => {
        expect(providerSafeToolName('web_search', declared)).toBe('web_search');
        expect(providerSafeToolName('mcp.github.search', declared)).toBe('mcp.github.search');
    });

    it('turns an empty or missing name into "_"', () => {
        expect(providerSafeToolName('', declared)).toBe('_');
        expect(providerSafeToolName(undefined, declared)).toBe('_');
        expect(providerSafeToolName(null, declared)).toBe('_');
    });

    it('replaces every rejected character of an undeclared name', () => {
        expect(providerSafeToolName('multi_tool_use.parallel', declared)).toBe('multi_tool_use_parallel');
        expect(providerSafeToolName('functions.x', declared)).toBe('functions_x');
        expect(providerSafeToolName('a b/c:d', declared)).toBe('a_b_c_d');
    });

    it('leaves an undeclared name that is already valid unchanged', () => {
        expect(providerSafeToolName('removed_tool-2', declared)).toBe('removed_tool-2');
    });

    it('truncates an undeclared name to the limit', () => {
        expect(providerSafeToolName('x'.repeat(100), declared)).toHaveLength(64);
        expect(providerSafeToolName('abcdef', declared, 3)).toBe('abc');
    });

    it('treats every name as undeclared when the request declares no tools', () => {
        expect(providerSafeToolName('functions.x', declaredToolNames(undefined))).toBe('functions_x');
    });
});
