import type { HttpTimeoutOptions } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { exposePrivate } from '../test/__helpers__/test-utils.js';
import { AnthropicDriver } from './anthropic/index.js';
import { BedrockDriver } from './bedrock/index.js';
import { GroqDriver } from './groq/index.js';
import { HuggingFaceIEDriver } from './huggingface_ie.js';
import { MistralAIDriver } from './mistral/index.js';
import { AzureOpenAIDriver } from './openai/azure_openai.js';
import { OpenAIDriver } from './openai/openai.js';
import { OpenAICompatibleDriver } from './openai/openai_compatible.js';
import { ReplicateDriver } from './replicate.js';
import { TogetherAIDriver } from './togetherai/index.js';
import { VertexAIDriver } from './vertexai/index.js';
import { WatsonxDriver } from './watsonx/index.js';
import { xAIDriver } from './xai/index.js';

type DriverFetchInternals = {
    _driverFetch?: typeof fetch;
};

type SdkFetchHolder = {
    fetch?: typeof fetch;
    _options?: {
        fetch?: typeof fetch;
    };
};

type FetchClientInternals = {
    _fetch: Promise<typeof fetch>;
};

type BedrockRequestHandlerConfig = {
    requestTimeout: number;
    throwOnRequestTimeout: boolean;
    connectionTimeout: number;
    socketTimeout: number;
};

function driverFetch(driver: object): typeof fetch {
    const fetchImpl = exposePrivate<DriverFetchInternals>(driver)._driverFetch;
    if (!fetchImpl) {
        throw new Error('Driver fetch was not initialized');
    }
    return fetchImpl;
}

function expectSdkUsesDriverFetch(driver: object, client: object): void {
    const expectedFetch = driverFetch(driver);
    const sdkClient = exposePrivate<SdkFetchHolder>(client);
    expect(sdkClient.fetch ?? sdkClient._options?.fetch).toBe(expectedFetch);
}

async function expectFetchClientUsesDriverFetch(driver: object, client: object): Promise<void> {
    await expect(exposePrivate<FetchClientInternals>(client)._fetch).resolves.toBe(driverFetch(driver));
}

describe('driver HTTP timeout wiring', () => {
    it('configures Bedrock request handlers to throw on request timeout', () => {
        const driver = new BedrockDriver({
            region: 'us-east-1',
            httpTimeout: {
                headersTimeout: 1_234,
                bodyTimeout: 2_345,
                connectTimeout: 345,
            },
        });

        const config = exposePrivate<{
            getBedrockRequestHandlerConfig: (httpTimeout?: HttpTimeoutOptions) => BedrockRequestHandlerConfig;
        }>(driver).getBedrockRequestHandlerConfig();

        expect(config).toEqual({
            requestTimeout: 1_234,
            throwOnRequestTimeout: true,
            connectionTimeout: 345,
            socketTimeout: 2_345,
        });

        const overrideConfig = exposePrivate<{
            getBedrockRequestHandlerConfig: (httpTimeout?: HttpTimeoutOptions) => BedrockRequestHandlerConfig;
        }>(driver).getBedrockRequestHandlerConfig({
            headersTimeout: 456,
            bodyTimeout: 567,
        });

        expect(overrideConfig).toEqual({
            requestTimeout: 456,
            throwOnRequestTimeout: true,
            connectionTimeout: 345,
            socketTimeout: 567,
        });

        const defaultExecutor = driver.getExecutor();
        const scopedExecutor = driver.getExecutor({ headersTimeout: 456 });
        expect(scopedExecutor).not.toBe(defaultExecutor);
        expect(driver.getExecutor()).toBe(defaultExecutor);
        scopedExecutor.destroy();
        driver.destroy();
    });

    it('uses a longer Bedrock timeout default for non-streaming text execution', () => {
        const driver = new BedrockDriver({
            region: 'us-east-1',
        });

        const internals = exposePrivate<{
            getBedrockRequestHandlerConfig: (httpTimeout?: HttpTimeoutOptions) => BedrockRequestHandlerConfig;
            getBedrockNonStreamingHttpTimeout: (httpTimeout?: HttpTimeoutOptions) => HttpTimeoutOptions;
        }>(driver);

        expect(internals.getBedrockRequestHandlerConfig()).toEqual({
            requestTimeout: 60_000,
            throwOnRequestTimeout: true,
            connectionTimeout: 10_000,
            socketTimeout: 60_000,
        });
        expect(internals.getBedrockRequestHandlerConfig(internals.getBedrockNonStreamingHttpTimeout())).toEqual({
            requestTimeout: 900_000,
            throwOnRequestTimeout: true,
            connectionTimeout: 10_000,
            socketTimeout: 900_000,
        });

        driver.destroy();
    });

    it('keeps explicit Bedrock non-streaming timeout overrides field-by-field', () => {
        const driver = new BedrockDriver({
            region: 'us-east-1',
            httpTimeout: {
                bodyTimeout: 120_000,
                connectTimeout: 1_000,
            },
        });

        const internals = exposePrivate<{
            getBedrockRequestHandlerConfig: (httpTimeout?: HttpTimeoutOptions) => BedrockRequestHandlerConfig;
            getBedrockNonStreamingHttpTimeout: (httpTimeout?: HttpTimeoutOptions) => HttpTimeoutOptions;
        }>(driver);

        expect(
            internals.getBedrockRequestHandlerConfig(
                internals.getBedrockNonStreamingHttpTimeout({
                    headersTimeout: 180_000,
                }),
            ),
        ).toEqual({
            requestTimeout: 180_000,
            throwOnRequestTimeout: true,
            connectionTimeout: 1_000,
            socketTimeout: 120_000,
        });

        driver.destroy();
    });

    it('passes the driver fetch to SDK clients that accept a custom fetch implementation', () => {
        const openai = new OpenAIDriver({ apiKey: 'test-key' });
        expectSdkUsesDriverFetch(openai, openai.service);
        openai.destroy();

        const azure = new AzureOpenAIDriver({
            apiKey: 'test-key',
            endpoint: 'https://example.openai.azure.com',
            deployment: 'deployment',
        });
        expectSdkUsesDriverFetch(azure, azure.service);
        azure.destroy();

        const compatible = new OpenAICompatibleDriver({
            apiKey: 'test-key',
            endpoint: 'https://example.test/v1',
        });
        expectSdkUsesDriverFetch(compatible, compatible.service);
        compatible.destroy();

        const together = new TogetherAIDriver({ apiKey: 'test-key' });
        expectSdkUsesDriverFetch(together, together.service);
        together.destroy();

        const anthropic = new AnthropicDriver({ apiKey: 'test-key' });
        expectSdkUsesDriverFetch(anthropic, anthropic.client);
        anthropic.destroy();

        const groq = new GroqDriver({ apiKey: 'test-key' });
        expectSdkUsesDriverFetch(groq, groq.client);
        groq.destroy();

        const replicate = new ReplicateDriver({ apiKey: 'test-key' });
        expectSdkUsesDriverFetch(replicate, replicate.service);
        replicate.destroy();

        const xai = new xAIDriver({ apiKey: 'test-key' });
        expectSdkUsesDriverFetch(xai, xai.service);
        xai.destroy();
    });

    it('passes the driver fetch to FetchClient-backed drivers', async () => {
        const mistral = new MistralAIDriver({ apiKey: 'test-key' });
        await expectFetchClientUsesDriverFetch(mistral, mistral.client);
        mistral.destroy();

        const watsonx = new WatsonxDriver({
            apiKey: 'test-key',
            projectId: 'project',
            endpointUrl: 'https://example.test',
        });
        await expectFetchClientUsesDriverFetch(watsonx, watsonx.fetchClient);
        watsonx.destroy();

        const huggingFace = new HuggingFaceIEDriver({
            apiKey: 'test-key',
            endpoint_url: 'https://example.test',
        });
        await expectFetchClientUsesDriverFetch(huggingFace, huggingFace.service);
        huggingFace.destroy();

        const xai = new xAIDriver({ apiKey: 'test-key' });
        await expectFetchClientUsesDriverFetch(xai, xai.xai_service);
        xai.destroy();
    });

    it('passes the driver fetch to Vertex AI FetchClient paths', async () => {
        const driver = new VertexAIDriver({
            project: 'project',
            region: 'us-central1',
        });

        await expectFetchClientUsesDriverFetch(driver, driver.getFetchClient());
        await expectFetchClientUsesDriverFetch(driver, driver.getLLamaClient());

        driver.destroy();
    });

    it('maps Vertex AI single-timeout SDK clients from DriverOptions.httpTimeout', () => {
        const driver = new VertexAIDriver({
            project: 'project',
            region: 'us-central1',
            httpTimeout: {
                headersTimeout: 1_000,
                bodyTimeout: 3_000,
            },
        });

        const internals = exposePrivate<{
            getGoogleGenAIHttpOptions: (
                flex: boolean,
                httpTimeout?: HttpTimeoutOptions,
            ) => {
                timeout: number;
                headers?: Record<string, string>;
            };
            getAnthropicVertexClientOptions: (
                region: string,
                authClient: unknown,
                httpTimeout?: HttpTimeoutOptions,
            ) => {
                timeout: number;
                region: string;
                projectId: string;
                fetch: typeof fetch;
            };
        }>(driver);

        expect(internals.getGoogleGenAIHttpOptions(false)).toEqual({
            timeout: 3_000,
        });
        expect(internals.getGoogleGenAIHttpOptions(true)).toEqual({
            timeout: 3_000,
            headers: {
                'X-Vertex-AI-LLM-Request-Type': 'shared',
                'X-Vertex-AI-LLM-Shared-Request-Type': 'flex',
            },
        });
        expect(
            internals.getGoogleGenAIHttpOptions(false, {
                headersTimeout: 10_000,
                bodyTimeout: 20_000,
            }),
        ).toEqual({
            timeout: 20_000,
        });

        const anthropicOptions = internals.getAnthropicVertexClientOptions('global', {});
        expect(anthropicOptions).toEqual({
            timeout: 3_000,
            region: 'global',
            projectId: 'project',
            authClient: {},
            fetch: driverFetch(driver),
        });
        expect(
            internals.getAnthropicVertexClientOptions(
                'global',
                {},
                {
                    headersTimeout: 10_000,
                    bodyTimeout: 20_000,
                },
            ),
        ).toEqual({
            timeout: 20_000,
            region: 'global',
            projectId: 'project',
            authClient: {},
            fetch: driverFetch(driver),
        });

        const defaultGoogleClient = driver.getGoogleGenAIClient();
        const overrideGoogleClient = driver.getGoogleGenAIClient('us-central1', false, {
            headersTimeout: 10_000,
        });
        expect(overrideGoogleClient).not.toBe(defaultGoogleClient);
        expect(driver.getGoogleGenAIClient()).toBe(defaultGoogleClient);

        driver.destroy();
    });

    it('allows ten minutes for Gemini requests by default', () => {
        const driver = new VertexAIDriver({
            project: 'project',
            region: 'us-central1',
        });
        const internals = exposePrivate<{
            getGoogleGenAIHttpOptions: (flex: boolean) => { timeout: number };
            getAnthropicVertexClientOptions: (region: string, authClient: unknown) => { timeout: number };
        }>(driver);
        const connectOnlyDriver = new VertexAIDriver({
            project: 'project',
            region: 'us-central1',
            httpTimeout: { connectTimeout: 1_000 },
        });
        const connectOnlyInternals = exposePrivate<{
            getGoogleGenAIHttpOptions: (flex: boolean) => { timeout: number };
        }>(connectOnlyDriver);

        expect(internals.getGoogleGenAIHttpOptions(false).timeout).toBe(600_000);
        expect(connectOnlyInternals.getGoogleGenAIHttpOptions(false).timeout).toBe(600_000);
        expect(internals.getAnthropicVertexClientOptions('global', {}).timeout).toBe(60_000);

        connectOnlyDriver.destroy();
        driver.destroy();
    });
});
