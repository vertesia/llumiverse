/**
 * The curated live smoke targets: one or two models per provider, enabled by that provider's credentials and
 * narrowed with LLUMIVERSE_LIVE_PROVIDERS / LLUMIVERSE_LIVE_MODELS.
 */
import type { AbstractDriver } from '@llumiverse/core';
import 'dotenv/config';
import { GoogleAuth } from 'google-auth-library';
import {
    AzureOpenAIDriver,
    BedrockDriver,
    BedrockMantleDriver,
    GroqDriver,
    MistralAIDriver,
    OpenAIDriver,
    OpenAIResponsesDriver,
    TogetherAIDriver,
    VertexAIDriver,
    WatsonxDriver,
    xAIDriver,
} from '../src/index.js';
import { selectLiveTestDrivers } from './live-model-selection.js';

export interface TestDriver {
    driver: AbstractDriver;
    models: string[];
    name: string;
}

const drivers: TestDriver[] = [];

if (process.env.GOOGLE_PROJECT_ID && process.env.GOOGLE_REGION) {
    new GoogleAuth();
    drivers.push({
        name: 'google-vertex',
        driver: new VertexAIDriver({
            project: process.env.GOOGLE_PROJECT_ID as string,
            region: process.env.GOOGLE_REGION as string,
        }),
        models: [
            'locations/global/publishers/google/models/gemini-3.5-flash-lite',
            'locations/global/publishers/anthropic/models/claude-haiku-5-5',
        ],
    });
} else {
    console.warn('Google Vertex tests are skipped: GOOGLE_PROJECT_ID environment variable is not set');
}

if (process.env.MISTRAL_API_KEY) {
    drivers.push({
        name: 'mistralai',
        driver: new MistralAIDriver({
            apiKey: process.env.MISTRAL_API_KEY as string,
            endpoint_url: (process.env.MISTRAL_ENDPOINT_URL as string) ?? undefined,
        }),
        models: ['mistral-small-latest'],
    });
} else {
    console.warn('MistralAI tests are skipped: MISTRAL_API_KEY environment variable is not set');
}

if (process.env.TOGETHER_API_KEY) {
    drivers.push({
        name: 'togetherai',
        driver: new TogetherAIDriver({
            apiKey: process.env.TOGETHER_API_KEY as string,
        }),
        // A cheap serverless reasoning model with a reasoning-effort control, so the smoke's low effort reaches it.
        // Qwen3.5-9B had none and sometimes reasoned until max_tokens without answering.
        models: ['zai-org/GLM-5.3-Flash'],
    });
} else {
    console.warn('TogetherAI tests are skipped: TOGETHER_API_KEY environment variable is not set');
}

if (process.env.OPENAI_API_KEY) {
    drivers.push({
        name: 'openai',
        driver: new OpenAIDriver({
            apiKey: process.env.OPENAI_API_KEY as string,
        }),
        models: ['gpt-6-luna'],
    });
} else {
    console.warn('OpenAI tests are skipped: OPENAI_API_KEY environment variable is not set');
}

// Azure OpenAI addresses models by deployment name; the deployment must exist on the configured resource.
const AZURE_OPENAI_MODELS = ['gpt-6-luna'];

if (process.env.AZURE_OPENAI_KEY && process.env.AZURE_OPENAI_ENDPOINT) {
    drivers.push({
        name: 'azure-openai-key',
        driver: new AzureOpenAIDriver({
            apiKey: process.env.AZURE_OPENAI_KEY as string,
            endpoint: process.env.AZURE_OPENAI_ENDPOINT as string,
            deployment: process.env.AZURE_OPENAI_DEPLOYMENT as string,
        }),
        models: AZURE_OPENAI_MODELS,
    });
} else {
    console.warn('Azure OpenAI tests are skipped: AZURE_OPENAI_KEY environment variable is not set');
}

if (process.env.BEDROCK_REGION) {
    drivers.push({
        name: 'bedrock',
        driver: new BedrockDriver({
            region: process.env.BEDROCK_REGION as string,
        }),
        //Use foundation models and inference profiles to test the driver
        // One global and one regional (us.) inference profile.
        models: ['global.anthropic.claude-haiku-5-5', 'us.amazon.nova-2-lite-v1:0'],
    });
} else {
    console.warn('Bedrock tests are skipped: BEDROCK_REGION environment variable is not set');
}

if (process.env.BEDROCK_MANTLE_REGION) {
    drivers.push({
        name: 'bedrock-mantle',
        driver: new BedrockMantleDriver({ region: process.env.BEDROCK_MANTLE_REGION }),
        // Mantle does not list Claude Haiku 5.5 yet; Haiku 4.5 is its smallest Claude model.
        // One model per Mantle protocol: Responses (gpt-6-luna), Chat Completions (gpt-oss-20b) and Messages (Claude).
        models: ['openai.gpt-6-luna', 'openai.gpt-oss-20b', 'anthropic.claude-haiku-4-5'],
    });
} else {
    console.warn('Bedrock Mantle tests are skipped: BEDROCK_MANTLE_REGION environment variable is not set');
}

if (process.env.GROQ_API_KEY) {
    drivers.push({
        name: 'groq',
        driver: new GroqDriver({
            apiKey: process.env.GROQ_API_KEY as string,
        }),
        // gpt-oss-20b is Groq's cheapest self-serve model; qwen3.8-27b covers Groq's Qwen output cap and effort levels.
        // Tool calling on Groq's gpt-oss is not covered here:
        // https://community.groq.com/t/gpt-oss-120b-ignoring-tools/385/29
        models: ['openai/gpt-oss-20b', 'qwen/qwen3.8-27b'],
    });
} else {
    console.warn('Groq tests are skipped: GROQ_API_KEY environment variable is not set');
}

if (process.env.WATSONX_API_KEY) {
    drivers.push({
        name: 'watsonx',
        driver: new WatsonxDriver({
            apiKey: process.env.WATSONX_API_KEY as string,
            projectId: process.env.WATSONX_PROJECT_ID as string,
            endpointUrl: process.env.WATSONX_ENDPOINT_URL as string,
        }),
        models: ['ibm/granite-4-h-small'],
    });
} else {
    console.warn('Watsonx tests are skipped: WATSONX_API_KEY environment variable is not set');
}

if (process.env.OPENROUTER_API_KEY) {
    drivers.push({
        name: 'openrouter',
        driver: new OpenAIResponsesDriver({
            apiKey: process.env.OPENROUTER_API_KEY,
            endpoint: 'https://openrouter.ai/api/v1',
        }),
        // qwen/qwen3.8-flash was dropped: its upstream on OpenRouter answered 429 on most runs, outlasting the retries.
        models: ['minimax/minimax-m3', 'deepseek/deepseek-v4-flash', 'google/gemini-3.5-flash-lite'],
    });
} else {
    console.warn('OpenRouter tests are skipped: OPENROUTER_API_KEY environment variable is not set');
}

if (process.env.XAI_API_KEY) {
    drivers.push({
        name: 'xai',
        driver: new xAIDriver({
            apiKey: process.env.XAI_API_KEY as string,
        }),
        models: ['grok-4.7'],
    });
} else {
    console.warn('xAI tests are skipped: XAI_API_KEY environment variable is not set');
}

export const selectedDrivers = selectLiveTestDrivers(drivers, {
    providers: process.env.LLUMIVERSE_LIVE_PROVIDERS,
    models: process.env.LLUMIVERSE_LIVE_MODELS,
});
