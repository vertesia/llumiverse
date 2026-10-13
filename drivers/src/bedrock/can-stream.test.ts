import type { Bedrock } from '@aws-sdk/client-bedrock';
import type { ExecutionOptions } from '@llumiverse/core';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from './index.js';

class StreamingCheckDriver extends BedrockDriver {
    checkCanStream(model: string): Promise<boolean> {
        return this.canStream({ model } as ExecutionOptions);
    }
}

function logger() {
    return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

describe('Bedrock canStream check', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('reports a failed lookup through the driver logger and does not stream', async () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
        const log = logger();
        const driver = new StreamingCheckDriver({ region: 'us-east-1', logger: log });
        const denied = new Error('not authorized to perform bedrock:GetFoundationModel');
        const service = {
            getFoundationModel: vi.fn().mockRejectedValue(denied),
        } as unknown as Bedrock;
        vi.spyOn(driver, 'getService').mockReturnValue(service);

        // The lookup result is cached per model, so use an id no other test resolves.
        const model = 'arn:aws:bedrock:eu-west-1::foundation-model/can-stream-check.denied-v1';
        await expect(driver.checkCanStream(model)).resolves.toBe(false);

        expect(log.warn).toHaveBeenCalledWith(
            { err: denied, model, region: 'eu-west-1' },
            'Error on canStream check; not streaming',
        );
        expect(warn).not.toHaveBeenCalled();
    });
});
