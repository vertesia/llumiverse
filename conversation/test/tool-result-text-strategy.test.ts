import { describe, expect, it } from 'vitest';
import { selectToolResultTextBlocks, type ToolResultTextStrategy } from '../src/tool-result-text-strategy.js';

const first = { turn_id: 'turn:result', result_block_id: 'result:one', block_id: 'text:first', text: 'α'.repeat(8192) };
const second = { ...first, block_id: 'text:second', text: 'β'.repeat(8192) };
const small = { ...first, block_id: 'text:small', text: 'unchanged' };
function strategy(configuration: ToolResultTextStrategy['configuration']): ToolResultTextStrategy {
    return { processor_id: 'externalize-tool-result-text', processor_version: '2', configuration };
}

describe('authenticated tool-result text part nomination', () => {
    it('chooses one exact sibling rather than all threshold-eligible text and preserves original order', () => {
        const chosen = strategy({
            selector: {
                kind: 'exact_blocks',
                blocks: [{ turn_id: first.turn_id, result_block_id: first.result_block_id, block_id: first.block_id }],
            },
        });
        expect(selectToolResultTextBlocks([first, second, small], chosen)).toEqual([first]);
        expect(
            selectToolResultTextBlocks(
                [first, second, small],
                strategy({
                    selector: { kind: 'minimum_text_bytes', minimum_text_bytes: 16384 },
                }),
            ),
        ).toEqual([first, second]);
        expect(
            selectToolResultTextBlocks(
                [first, second, small],
                strategy({
                    selector: { kind: 'all_text' },
                }),
            ),
        ).toEqual([first, second, small]);
    });
    it.each([
        { turn_id: 'turn:foreign', result_block_id: first.result_block_id, block_id: first.block_id },
        { turn_id: first.turn_id, result_block_id: 'result:foreign', block_id: first.block_id },
        { turn_id: first.turn_id, result_block_id: first.result_block_id, block_id: 'json:original' },
    ])('rejects unavailable or nontext nominations without changing candidates', (block) => {
        const candidates = structuredClone([first, second]);
        expect(() =>
            selectToolResultTextBlocks(
                candidates,
                strategy({
                    selector: { kind: 'exact_blocks', blocks: [block] },
                }),
            ),
        ).toThrow('unknown or nontext accepted content');
        expect(candidates).toEqual([first, second]);
    });
    it('rejects duplicate nominations and ambiguous authenticated identities', () => {
        const block = { turn_id: first.turn_id, result_block_id: first.result_block_id, block_id: first.block_id };
        expect(() =>
            selectToolResultTextBlocks(
                [first],
                strategy({
                    selector: { kind: 'exact_blocks', blocks: [block, block] },
                }),
            ),
        ).toThrow('repeats an accepted block');
        expect(() => selectToolResultTextBlocks([first, first])).toThrow('ambiguous');
    });
    it('retains v1 empty configuration semantics and rejects v2 configuration in v1', () => {
        const original = { processor_id: 'externalize-tool-result-text', processor_version: '1', configuration: {} };
        expect(selectToolResultTextBlocks([first, second], original)).toEqual([first, second]);
        expect(() =>
            selectToolResultTextBlocks([first], { ...original, configuration: { selector: { kind: 'all_text' } } }),
        ).toThrow('v1 requires its original empty configuration');
    });
});
