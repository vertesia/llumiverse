/**
 * Reproduction test for the non-retryable Vertex/Anthropic 400 that terminates
 * agent conversations:
 *
 *   messages.N.content.k: unexpected `tool_use_id` found in `tool_result` blocks: <id>.
 *   Each `tool_result` block must have a corresponding `tool_use` block in the
 *   previous message.
 *
 * Root cause: a `tool_result` survives in the conversation while its matching
 * `tool_use` does not appear in the immediately-preceding assistant message —
 * e.g. conversation compaction trims the `tool_use` turn but keeps its result,
 * or a parallel tool batch's results land in a message that can't be re-paired.
 *
 * This drives the real send-prep (`getClaudePayload`) end-to-end and asserts the
 * produced payload contains no orphaned `tool_result` (the exact condition the
 * API validates). It fails against the unfixed pipeline and passes once
 * `fixOrphanedToolResults` is wired in.
 */

import type { MessageParam } from '@anthropic-ai/sdk/resources/index.js';
import type { ExecutionOptions } from '@llumiverse/core';
import { describe, expect, test } from 'vitest';
import { type ClaudePrompt, getClaudePayload } from '../src/shared/claude-messages.js';

/**
 * Returns the tool_use_ids of any `tool_result` blocks whose matching `tool_use`
 * is not declared by the immediately-preceding assistant message — i.e. exactly
 * what the Anthropic/Vertex API rejects with a 400.
 */
function findOrphanedToolResults(messages: MessageParam[]): string[] {
    const orphans: string[] = [];
    for (let i = 0; i < messages.length; i++) {
        const message = messages[i];
        if (message.role !== 'user' || !Array.isArray(message.content)) continue;
        const prev = messages[i - 1];
        const allowed = new Set<string>();
        if (prev && prev.role === 'assistant' && Array.isArray(prev.content)) {
            for (const block of prev.content) {
                if (block.type === 'tool_use') allowed.add(block.id);
            }
        }
        for (const block of message.content) {
            if (block.type === 'tool_result' && !allowed.has(block.tool_use_id)) {
                orphans.push(block.tool_use_id);
            }
        }
    }
    return orphans;
}

// Tools must be present so getClaudePayload preserves tool blocks instead of
// converting them to text (which would hide the bug).
const OPTIONS = {
    model: 'claude-sonnet-4-6',
    tools: [{ name: 'launch_workstream', description: '', input_schema: { type: 'object', properties: {} } }],
} as unknown as ExecutionOptions;

describe('getClaudePayload — orphaned tool_result (Vertex 400 repro)', () => {
    test('a tool_result whose tool_use was dropped does not reach the payload', () => {
        // Mirrors the real failed conversation: a compaction summary, a valid
        // tool turn, then a parallel-result message where one result's tool_use
        // is absent (its assistant tool_use was compacted away).
        const prompt: ClaudePrompt = {
            system: undefined,
            messages: [
                { role: 'user', content: [{ type: 'text', text: '[checkpoint summary of 454 messages]' }] },
                { role: 'assistant', content: [{ type: 'tool_use', id: 'think_1', name: 'think', input: {} }] },
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'think_1', content: 'ok' }] },
                {
                    role: 'assistant',
                    content: [
                        { type: 'text', text: 'Launching workstream.' },
                        { type: 'tool_use', id: 'launch_1', name: 'launch_workstream', input: {} },
                    ],
                },
                {
                    role: 'user',
                    content: [
                        { type: 'tool_result', tool_use_id: 'launch_1', content: 'started' },
                        // Orphan: matching tool_use is gone (compacted away).
                        { type: 'tool_result', tool_use_id: 'toolu_orphan_gone', content: 'dangling result' },
                    ],
                },
            ],
        };

        const { payload } = getClaudePayload(OPTIONS, prompt);
        const orphans = findOrphanedToolResults(payload.messages as MessageParam[]);
        expect(orphans).toEqual([]);

        // The valid result is retained.
        expect(JSON.stringify(payload.messages)).toContain('launch_1');
    });

    test('parallel tool results split across user messages stay paired after prep', () => {
        // Two parallel tool_uses whose results were stored as two separate user
        // messages. Merge must recombine them under the assistant turn so neither
        // is treated as an orphan.
        const prompt: ClaudePrompt = {
            system: undefined,
            messages: [
                {
                    role: 'assistant',
                    content: [
                        { type: 'tool_use', id: 'get_project', name: 'get_project', input: {} },
                        { type: 'tool_use', id: 'learn_ops', name: 'learn_artifact_operations', input: {} },
                    ],
                },
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'get_project', content: 'p' }] },
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'learn_ops', content: 'o' }] },
            ],
        };

        const { payload } = getClaudePayload(OPTIONS, prompt);
        expect(findOrphanedToolResults(payload.messages as MessageParam[])).toEqual([]);
        // Both legitimate results survive.
        const json = JSON.stringify(payload.messages);
        expect(json).toContain('get_project');
        expect(json).toContain('learn_ops');
    });
});
