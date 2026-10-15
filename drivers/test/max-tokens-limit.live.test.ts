/**
 * Checks that each smoke model accepts max_tokens at its documented output limit. Run it when a model's limit
 * changes in the model catalog; it is live but deliberately not in CI's allow list. Claude providers count max_tokens
 * against the output-tokens-per-minute quota when a request starts, so a call at the full limit can use most of a
 * minute's quota and push the smoke tests running beside it into rate limits.
 */
import {
    getMaxOutputTokens,
    getMaxTokensLimitBedrock,
    getMaxTokensLimitVertexAi,
    PromptRole,
    type PromptSegment,
    type Providers,
    resolveModelProfile,
} from '@llumiverse/core';
import { describe, expect, test } from 'vitest';
import { selectedDrivers } from './live-smoke-drivers.js';
import { getLowEffortTestOptions, retryUnlessRateLimited, withRateLimitBackoff } from './utils.js';

const TIMEOUT = 90 * 1000;

// Drivers run in parallel; one model's requests stay in order so a throttled model is not sent several at once.
describe.concurrent.each(selectedDrivers)('Driver $name', ({ name, driver, models }) => {
    describe.each(models)('%s', (model) => {
        test(`${name}: max_tokens at documented limit on ${model}`, {
            timeout: TIMEOUT,
            retry: retryUnlessRateLimited(1),
            concurrent: false,
        }, async (context) => {
            // Use provider-specific limits where available; unknown Vertex limits must remain unknown.
            let limit: number | undefined;
            if (driver.provider === 'bedrock') {
                limit = getMaxTokensLimitBedrock(model);
            } else if (driver.provider === 'vertexai') {
                limit = getMaxTokensLimitVertexAi(model);
                if (limit === undefined) {
                    context.skip();
                    return;
                }
            }
            // Other providers: the provider-aware profile (e.g. Groq's own completion caps), else the canonical limit
            if (!limit) {
                limit =
                    resolveModelProfile(model, driver.provider as Providers).max_output_tokens ??
                    getMaxOutputTokens(model);
            }

            const shortPrompt: PromptSegment[] = [{ role: PromptRole.user, content: 'Say "ok".' }];
            const options = getLowEffortTestOptions(model, driver.provider, { max_tokens: limit });
            // Guard against the advertised-options filter dropping the option this test is about.
            expect((options.model_options as { max_tokens?: number } | undefined)?.max_tokens).toBe(limit);
            const r = await withRateLimitBackoff(() => driver.execute(shortPrompt, options));
            // If the provider rejects our limit value, r.error will be set
            expect(r.error).toBeFalsy();
            expect(r.finish_reason).toBeTruthy();
        });
    });
});
