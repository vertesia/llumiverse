/**
 * Live Claude message threads (`message_thread: true`).
 *
 * A tool loop mixing execute() and stream() turns, with the conversation stored as JSON between turns. Every turn
 * after the first must continue the thread: a rejected continue falls back to a full request and logs it, which
 * these tests report as a failure, since the fallback alone would still produce correct answers.
 *
 * Runs with ANTHROPIC_API_KEY (Claude API) and, for Foundry, AZURE_FOUNDRY_ENDPOINT plus
 * AZURE_FOUNDRY_CLAUDE_MODEL (a Claude deployment on that endpoint).
 */

import type { Logger } from '@llumiverse/common';
import { type AbstractDriver, type ExecutionOptions, PromptRole, type PromptSegment } from '@llumiverse/core';
import 'dotenv/config';
import { describe, expect, test } from 'vitest';
import { AnthropicDriver, AzureFoundryDriver } from '../src/index.js';
import { answerText } from './assertions.js';
import { getLowEffortTestOptions, retryUnlessRateLimited, withRateLimitBackoff } from './utils.js';

const TIMEOUT = 180 * 1000;

interface ThreadLogs {
    fallbacks: unknown[];
}

interface TestDriver {
    name: string;
    model: string;
    create: (logger: Logger) => AbstractDriver;
}

const drivers: TestDriver[] = [];

if (process.env.ANTHROPIC_API_KEY) {
    drivers.push({
        name: 'anthropic',
        model: 'claude-haiku-5-5',
        create: (logger) => new AnthropicDriver({ apiKey: process.env.ANTHROPIC_API_KEY as string, logger }),
    });
} else {
    console.warn('Anthropic message thread tests are skipped: ANTHROPIC_API_KEY environment variable is not set');
}

if (process.env.AZURE_FOUNDRY_ENDPOINT && process.env.AZURE_FOUNDRY_CLAUDE_MODEL) {
    drivers.push({
        name: 'azure-foundry',
        model: process.env.AZURE_FOUNDRY_CLAUDE_MODEL,
        create: (logger) =>
            new AzureFoundryDriver({
                endpoint: process.env.AZURE_FOUNDRY_ENDPOINT as string,
                apiKey: process.env.AZURE_FOUNDRY_API_KEY || undefined,
                logger,
            }),
    });
} else {
    console.warn(
        'Foundry message thread tests are skipped: AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_CLAUDE_MODEL are not set',
    );
}

/** A logger that records thread fallbacks and drops everything else. */
function threadLogger(logs: ThreadLogs): Logger {
    const noop = () => undefined;
    const debug = (obj: unknown, msg?: string) => {
        if (typeof msg === 'string' && msg.includes('Message thread request rejected')) logs.fallbacks.push(obj);
    };
    return { debug, info: noop, warn: noop, error: noop } as Logger;
}

interface ThreadMeta {
    claude_thread?: { previous_message_id: string; stored_messages: number };
    claude_thread_disabled?: boolean;
}

function threadMeta(conversation: unknown): ThreadMeta {
    return ((conversation as { _llumiverse_meta?: ThreadMeta } | undefined)?._llumiverse_meta ?? {}) as ThreadMeta;
}

const TOOL_DEF = {
    name: 'get_locker_code',
    description: 'Get the code of a locker by its number',
    input_schema: {
        type: 'object' as const,
        properties: { locker: { type: 'integer', description: 'The locker number' } },
        required: ['locker'],
    },
};

async function streamTurn(driver: AbstractDriver, prompt: PromptSegment[], options: ExecutionOptions) {
    const stream = await driver.stream(prompt, options);
    for await (const _chunk of stream) {
        // Consume the stream so the completion is populated
    }
    if (!stream.completion) throw new Error('Stream ended without a completion');
    return stream.completion;
}

describe.skipIf(drivers.length > 0)('Claude message threads (no drivers configured)', () => {
    test('skipped', () => {
        console.warn('Claude message thread tests skipped: no Claude API or Foundry credentials');
    });
});

describe.skipIf(drivers.length === 0).concurrent.each(drivers)(
    'Claude message threads: $name',
    ({ name, model, create }) => {
        test(`${name}: a tool loop continues one thread across execute and stream turns`, {
            timeout: TIMEOUT,
            retry: retryUnlessRateLimited(1),
        }, async () => {
            const logs: ThreadLogs = { fallbacks: [] };
            const driver = create(threadLogger(logs));
            const options: ExecutionOptions = {
                ...getLowEffortTestOptions(model, driver.provider, { max_tokens: 1000 }),
                message_thread: true,
                tools: [TOOL_DEF],
            };
            const messageIds: string[] = [];
            const storedCounts: number[] = [];
            let conversation: unknown;
            const record = (result: { conversation?: unknown }) => {
                // Store and reload the conversation as an application would between turns.
                conversation = JSON.parse(JSON.stringify(result.conversation));
                const meta = threadMeta(conversation);
                expect(meta.claude_thread_disabled, `[${name}] message threads were turned off`).toBeFalsy();
                expect(
                    meta.claude_thread,
                    `[${name}] no thread state after turn ${messageIds.length + 1}`,
                ).toBeDefined();
                messageIds.push(meta.claude_thread?.previous_message_id ?? '');
                storedCounts.push(meta.claude_thread?.stored_messages ?? 0);
            };

            // Turn 1 (execute, create): the model calls the tool.
            const turn1 = await withRateLimitBackoff(() =>
                driver.execute(
                    [
                        { role: PromptRole.system, content: 'Use the tools when needed. Answer briefly.' },
                        { role: PromptRole.user, content: 'What is the code of locker 7?' },
                    ],
                    { ...options, conversation },
                ),
            );
            const toolUse = turn1.tool_use?.[0];
            if (!toolUse) throw new Error(`[${name}] Expected a tool call on turn 1`);
            expect(toolUse.tool_name).toBe('get_locker_code');
            record(turn1);

            // Turn 2 (stream, continue): the tool result.
            const turn2 = await withRateLimitBackoff(() =>
                streamTurn(driver, [{ role: PromptRole.tool, tool_use_id: toolUse.id, content: 'QX-4471' }], {
                    ...options,
                    conversation,
                }),
            );
            expect(answerText(turn2)).toContain('QX-4471');
            record(turn2);

            // Turn 3 (execute, continue): only the thread holds the code, as the new request sends one message.
            const turn3 = await withRateLimitBackoff(() =>
                driver.execute(
                    [{ role: PromptRole.user, content: 'Repeat the code of locker 7, without calling a tool.' }],
                    { ...options, conversation },
                ),
            );
            expect(answerText(turn3)).toContain('QX-4471');
            record(turn3);

            // Turn 4 (stream, continue).
            const turn4 = await withRateLimitBackoff(() =>
                streamTurn(
                    driver,
                    [{ role: PromptRole.user, content: 'Which locker number did I ask about? Digits only.' }],
                    { ...options, conversation },
                ),
            );
            expect(answerText(turn4)).toContain('7');
            record(turn4);

            expect(logs.fallbacks, `[${name}] thread requests fell back to full requests`).toEqual([]);
            expect(new Set(messageIds).size, `[${name}] each turn must store a new message id`).toBe(4);
            for (let i = 1; i < storedCounts.length; i++) {
                expect(
                    storedCounts[i],
                    `[${name}] stored messages must grow: ${storedCounts.join(', ')}`,
                ).toBeGreaterThan(storedCounts[i - 1]);
            }
        });

        test(`${name}: a conversation without the option sends no thread`, {
            timeout: TIMEOUT,
            retry: retryUnlessRateLimited(1),
        }, async () => {
            const logs: ThreadLogs = { fallbacks: [] };
            const driver = create(threadLogger(logs));
            const options = getLowEffortTestOptions(model, driver.provider, { max_tokens: 200 });
            const result = await withRateLimitBackoff(() =>
                driver.execute([{ role: PromptRole.user, content: 'Reply with exactly: OK' }], options),
            );
            expect(answerText(result)).toContain('OK');
            expect(threadMeta(result.conversation).claude_thread).toBeUndefined();
        });
    },
);
