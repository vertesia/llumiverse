import { Bedrock, type GetFoundationModelCommandInput } from '@aws-sdk/client-bedrock';
import type { ExecutionOptions } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from './index.js';

class DiscoveryDriver extends BedrockDriver {
    supportsStreaming(model: string, signal?: AbortSignal): Promise<boolean> {
        return this.canStream({ model } as ExecutionOptions, signal);
    }
}

function discovery(region: string, supportsStreaming: boolean) {
    const driver = new DiscoveryDriver({ region });
    const service = new Bedrock({
        region,
        credentials: { accessKeyId: 'test', secretAccessKey: 'test' },
    });
    const lookup = vi.fn(async (_input: GetFoundationModelCommandInput) => ({
        $metadata: {},
        modelDetails: { responseStreamingSupported: supportsStreaming },
    }));
    Reflect.set(service, 'getFoundationModel', lookup);
    const getService = vi.spyOn(driver, 'getService').mockReturnValue(service);
    return { driver, lookup, getService };
}

describe('Bedrock streaming discovery', () => {
    it('looks up a regionless foundation-model reference by its bare ID', async () => {
        const { driver, lookup, getService } = discovery('us-east-1', true);
        const modelId = 'anthropic.claude-haiku-discovery-v1:0';

        await expect(driver.supportsStreaming(`arn:aws:bedrock:::foundation-model/${modelId}`)).resolves.toBe(true);

        expect(getService).toHaveBeenCalledWith('us-east-1');
        expect(lookup).toHaveBeenCalledWith({ modelIdentifier: modelId }, undefined);
    });

    it('preserves an explicit regional foundation-model ARN', async () => {
        const { driver, lookup, getService } = discovery('us-east-1', true);
        const model = 'arn:aws-us-gov:bedrock:us-gov-west-1::foundation-model/anthropic.claude-regional-v1:0';

        await expect(driver.supportsStreaming(model)).resolves.toBe(true);

        expect(getService).toHaveBeenCalledWith('us-gov-west-1');
        expect(lookup).toHaveBeenCalledWith({ modelIdentifier: model }, undefined);
    });

    it('does not share portable model capability results across regions', async () => {
        const east = discovery('us-east-1', false);
        const west = discovery('us-west-2', true);
        const model = 'arn:aws:bedrock:::foundation-model/anthropic.claude-regional-cache-v1:0';

        await expect(east.driver.supportsStreaming(model)).resolves.toBe(false);
        await expect(west.driver.supportsStreaming(model)).resolves.toBe(true);

        expect(east.lookup).toHaveBeenCalledOnce();
        expect(west.lookup).toHaveBeenCalledOnce();
        await expect(west.driver.supportsStreaming(model)).resolves.toBe(true);
        expect(west.lookup).toHaveBeenCalledOnce();
    });

    it('propagates cancellation without caching it as unsupported streaming', async () => {
        const { driver, lookup } = discovery('us-east-1', true);
        const controller = new AbortController();
        const reason = new Error('cancelled discovery');
        const model = 'arn:aws:bedrock:::foundation-model/anthropic.claude-cancelled-discovery-v1:0';
        lookup.mockImplementationOnce(async () => {
            controller.abort(reason);
            throw reason;
        });

        await expect(driver.supportsStreaming(model, controller.signal)).rejects.toBe(reason);
        await expect(driver.supportsStreaming(model)).resolves.toBe(true);
        expect(lookup).toHaveBeenCalledTimes(2);
    });
});
