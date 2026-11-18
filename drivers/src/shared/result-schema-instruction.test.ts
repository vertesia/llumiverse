import { describe, expect, it } from 'vitest';
import {
    buildResultSchemaInstruction,
    isResultSchemaInstruction,
    mergeResultSchemaInstructions,
} from './result-schema-instruction.js';

type Message = { role: string; text: string };
const instructionText = (message: Message) =>
    message.role === 'system' && isResultSchemaInstruction(message.text) ? message.text : undefined;
const copy = (schema: unknown): Message => ({ role: 'system', text: buildResultSchemaInstruction(schema) });
const user = (text: string): Message => ({ role: 'user', text });

describe('mergeResultSchemaInstructions', () => {
    const a = { type: 'object', properties: { a: { type: 'string' } } };
    const b = { type: 'object', properties: { b: { type: 'string' } } };

    it('drops an incoming copy identical to a stored one and keeps the stored position', () => {
        const history = [copy(a), user('one'), { role: 'assistant', text: '{}' }];
        expect(mergeResultSchemaInstructions(history, [copy(a), user('two')], instructionText)).toEqual([
            ...history,
            user('two'),
        ]);
    });

    it('replaces stored copies when the schema changes', () => {
        const history = [copy(a), user('one')];
        expect(mergeResultSchemaInstructions(history, [copy(b), user('two')], instructionText)).toEqual([
            user('one'),
            copy(b),
            user('two'),
        ]);
    });

    it('still leaves one copy when the stored copy was truncated', () => {
        const truncated = { role: 'system', text: `${buildResultSchemaInstruction(a).slice(0, 120)}… [truncated]` };
        const merged = mergeResultSchemaInstructions([truncated, user('one')], [copy(a), user('two')], instructionText);
        expect(merged.filter((message) => instructionText(message) !== undefined)).toEqual([copy(a)]);
    });

    it('leaves history alone when the new prompt carries no copy', () => {
        const history = [copy(a), user('one')];
        expect(mergeResultSchemaInstructions(history, [user('two')], instructionText)).toEqual([
            ...history,
            user('two'),
        ]);
    });

    it('ignores copies the caller does not classify, such as suffix copies in user messages', () => {
        const suffix = user(buildResultSchemaInstruction(a));
        expect(mergeResultSchemaInstructions([suffix], [suffix], instructionText)).toEqual([suffix, suffix]);
    });
});
