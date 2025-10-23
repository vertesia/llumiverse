import { AIModel, AbstractDriver, ExecutionOptions, Modalities } from '@llumiverse/core';
import 'dotenv/config';
import { describe, expect, test } from "vitest";
import { AzureOpenAIDriver, BedrockDriver, GroqDriver, MistralAIDriver, OpenAIDriver, TogetherAIDriver, VertexAIDriver, WatsonxDriver, xAIDriver } from '../src';
import { assertCompletionOk, assertStreamingCompletionOk } from './assertions';
import { testPrompt_color, testPrompt_describeImage, testSchema_animalDescription, testSchema_color } from './samples';

const TIMEOUT = 90 * 1000;

interface TestDriver {
    driver: AbstractDriver;
    models: string[];
    name: string;
}

const drivers: TestDriver[] = [];


if (process.env.GOOGLE_PROJECT_ID && process.env.GOOGLE_REGION) {
    drivers.push({
        name: "google-vertex",
        driver: new VertexAIDriver({
            project: process.env.GOOGLE_PROJECT_ID as string,
            region: process.env.GOOGLE_REGION as string,
        }),
        models: [
            "publishers/google/models/gemini-2.5-flash-lite",
            "publishers/anthropic/models/claude-3-7-sonnet",
            "publishers/meta/models/llama-4-scout-17b-16e-instruct-maas"
        ]
    })
} else {
    console.warn("Google Vertex tests are skipped: GOOGLE_PROJECT_ID environment variable is not set");
}


if (process.env.MISTRAL_API_KEY) {
    drivers.push({
        name: "mistralai",
        driver: new MistralAIDriver({
            apiKey: process.env.MISTRAL_API_KEY as string,
            endpoint_url: process.env.MISTRAL_ENDPOINT_URL as string ?? undefined
        }),
        models: [
            "pixtral-large-latest",
            "mistral-small-latest",
            "mistral-large-latest"
        ]
    }
    )
} else {
    console.warn("MistralAI tests are skipped: MISTRAL_API_KEY environment variable is not set");
}

if (process.env.TOGETHER_API_KEY) {
    drivers.push({
        name: "togetherai",
        driver: new TogetherAIDriver({
            apiKey: process.env.TOGETHER_API_KEY as string
        }),
        models: [
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            //"mistralai/Mixtral-8x7B-Instruct-v0.1" //too slow in tests for now
        ]
    }
    )
} else {
    console.warn("TogetherAI tests are skipped: TOGETHER_API_KEY environment variable is not set");
}

if (process.env.OPENAI_API_KEY) {
    drivers.push({
        name: "openai",
        driver: new OpenAIDriver({
            apiKey: process.env.OPENAI_API_KEY as string
        }),
        models: [
            "o3-mini",
            "gpt-4o-mini",
            "o1-mini",
        ]
    }
    )
} else {
    console.warn("OpenAI tests are skipped: OPENAI_API_KEY environment variable is not set");
}

const AZURE_OPENAI_MODELS = [
    "gpt-4",
]


if (process.env.AZURE_OPENAI_KEY && process.env.AZURE_OPENAI_ENDPOINT) {
    drivers.push({
        name: "azure-openai-key",
        driver: new AzureOpenAIDriver({
            apiKey: process.env.AZURE_OPENAI_KEY as string,
            endpoint: process.env.AZURE_OPENAI_ENDPOINT as string,
            deployment: process.env.AZURE_OPENAI_DEPLOYMENT as string
        }),
        models: AZURE_OPENAI_MODELS
    })
} else {
    console.warn("Azure OpenAI tests are skipped: AZURE_OPENAI_KEY environment variable is not set");
}


if (process.env.BEDROCK_REGION) {
    drivers.push({
        name: "bedrock",
        driver: new BedrockDriver({
            region: process.env.BEDROCK_REGION as string,
        }),
        //Use foundation models and inference profiles to test the driver
        models: [
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us.meta.llama3-3-70b-instruct-v1:0",
            "us.amazon.nova-micro-v1:0",
            "ai21.jamba-1-5-mini-v1:0",
        ],
    });
} else {
    console.warn("Bedrock tests are skipped: BEDROCK_REGION environment variable is not set");
}

if (process.env.GROQ_API_KEY) {

    drivers.push({
        name: "groq",
        driver: new GroqDriver({
            apiKey: process.env.GROQ_API_KEY as string
        }),
        models: [
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile"
        ]
    })
} else {
    console.warn("Groq tests are skipped: GROQ_API_KEY environment variable is not set");
}


if (process.env.WATSONX_API_KEY) {

    drivers.push({
        name: "watsonx",
        driver: new WatsonxDriver({
            apiKey: process.env.WATSONX_API_KEY as string,
            projectId: process.env.WATSONX_PROJECT_ID as string,
            endpointUrl: process.env.WATSONX_ENDPOINT_URL as string
        }),
        models: [
            "ibm/granite-3-2-8b-instruct",
            "ibm/granite-3-2b-instruct",
            //"ibm/granite-13b-instruct-v2" Does not work with schemas
            //Non-ibm models don't work, may be account related
            //"meta/llama-3-3-70b-instruct",
            //"mistralai/pixtral-12b",
        ]
    })
} else {
    console.warn("Watsonx tests are skipped: WATSONX_API_KEY environment variable is not set");
}

if (process.env.XAI_API_KEY) {
    drivers.push({
        name: "xai",
        driver: new xAIDriver({
            apiKey: process.env.XAI_API_KEY as string,
        }),
        models: [
            "grok-3-beta",
            "grok-2-vision-1212",
        ]
    })
} else {
    console.warn("xAI tests are skipped: XAI_API_KEY environment variable is not set");
}

function getTestOptions(model: string): ExecutionOptions {
    if (model == "o1-mini" || model == "o3-mini") {
        return {
            model: model,
            model_options: {
                _option_id: "openai-thinking",
                max_tokens: 3000,
                stop_sequence: model == "o3-mini" ? ["haemoglobin"] : undefined,
            },
            output_modality: Modalities.text,
        };
    }

    return {
        model: model,
        model_options: {
            _option_id: "text-fallback",
            max_tokens: 128,
            temperature: 0.3,
            top_k: 40,
            top_p: 0.7,             //Some models do not support top_p = 1.0, set to 0.99 or lower.
            //   top_logprobs: 5,        //Currently not supported, option will be ignored
            presence_penalty: 0.1,      //Cohere Command R does not support using presence & frequency penalty at the same time
            frequency_penalty: -0.1,
            stop_sequence: ["haemoglobin"],
        },
        output_modality: Modalities.text,
    };
}

describe.concurrent.each(drivers)("Driver $name", ({ name, driver, models }) => {

    let fetchedModels: AIModel[];

    test(`${name}: list models`, { timeout: TIMEOUT, retry: 1 }, async () => {
        const r = await driver.listModels();
        fetchedModels = r;
        console.log(r)
        expect(r.length).toBeGreaterThan(0);
    });

    test.each(models)(`${name}: prompt generation for %s`, {}, async (model) => {
        const p = await driver.createPrompt(testPrompt_color, { model })
        expect(p).toBeDefined();
    });

    test.each(models)(`${name}: prompt generation for %s`, {}, async (model) => {
        const p = await driver.createPrompt(testPrompt_color, { model, result_schema: testSchema_color })
        expect(p).toBeDefined();
    });

    test.each(models)(`${name}: multimodal prompt generation for %s`, {}, async (model) => {
        const p = await driver.createPrompt(testPrompt_describeImage, { model })
        expect(p).toBeDefined();
    });

    test.each(models)(`${name}: execute prompt on %s`, { timeout: TIMEOUT, retry: 2 }, async (model) => {
        const r = await driver.execute(testPrompt_color, getTestOptions(model));
        console.log("Result for execute " + model, JSON.stringify(r));
        assertCompletionOk(r, model, driver);
    });

    test.each(models)(`${name}: execute prompt with streaming on %s`, { timeout: TIMEOUT, retry: 2 }, async (model) => {
        const r = await driver.stream(testPrompt_color, getTestOptions(model));
        const out = await assertStreamingCompletionOk(r);
        console.log("Result for streaming " + model, JSON.stringify(out));
    });

    test.each(models)(`${name}: execute prompt with schema on %s`, { timeout: TIMEOUT, retry: 2 }, async (model) => {
        const r = await driver.execute(testPrompt_color, { ...getTestOptions(model), result_schema: testSchema_color });
        console.log("Result for execute with schema " + model, JSON.stringify(r.result));
        assertCompletionOk(r, model, driver);
    });

    test.each(models)(`${name}: execute prompt with streaming and schema on %s`, { timeout: TIMEOUT, retry: 2 }, async (model) => {
        const r = await driver.stream(testPrompt_color, { ...getTestOptions(model), result_schema: testSchema_color });
        const out = await assertStreamingCompletionOk(r, true);
        console.log("Result for streaming with schema " + model, JSON.stringify(out));
    });


    test.each(models)(`${name}: multimodal test - describe image with %s`, { timeout: TIMEOUT, retry: 2 }, async (model) => {

        if (!fetchedModels) {
            fetchedModels = await driver.listModels();
        }

        const isMultiModal = fetchedModels?.find(r => r.id === model)?.is_multimodal;

        console.log(`${model} is multimodal: ` + isMultiModal)
        if (!isMultiModal) return;

        const r = await driver.execute(testPrompt_describeImage, {
            output_modality: Modalities.text,
            model: model,
            model_options: {
                _option_id: "text-fallback",
                temperature: 0.5,
                max_tokens: 1024,
            },
            result_schema: testSchema_animalDescription
        })
        console.log("Result", r)
        assertCompletionOk(r);
    });
});
