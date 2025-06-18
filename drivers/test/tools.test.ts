import { AIModel, AbstractDriver, ExecutionOptions, Modalities, PromptRole, PromptSegment } from '@llumiverse/core';
import 'dotenv/config';
import { GoogleAuth } from 'google-auth-library';
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
    const auth = new GoogleAuth();
    const client = auth.getClient();
    drivers.push({
        name: "google-vertex",
        driver: new VertexAIDriver({
            project: process.env.GOOGLE_PROJECT_ID as string,
            region: process.env.GOOGLE_REGION as string,
        }),
        models: [
            "gemini-1.5-pro",
            "publishers/anthropic/models/claude-3-7-sonnet",
        ]
    })
}


// if (process.env.MISTRAL_API_KEY) {
//     drivers.push({
//         name: "mistralai",
//         driver: new MistralAIDriver({
//             apiKey: process.env.MISTRAL_API_KEY as string,
//             endpoint_url: process.env.MISTRAL_ENDPOINT_URL as string ?? undefined
//         }),
//         models: [
//             "open-mixtral-8x7b",
//             "mistral-medium-latest",
//             "mistral-large-latest"
//         ]
//     }
//     )
// } else {
//     console.warn("MistralAI tests are skipped: MISTRAL_API_KEY environment variable is not set");
// }

// if (process.env.TOGETHER_API_KEY) {
//     drivers.push({
//         name: "togetherai",
//         driver: new TogetherAIDriver({
//             apiKey: process.env.TOGETHER_API_KEY as string
//         }),
//         models: [
//             "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
//             //"mistralai/Mixtral-8x7B-Instruct-v0.1" //too slow in tests for now
//         ]
//     }
//     )
// } else {
//     console.warn("TogetherAI tests are skipped: TOGETHER_API_KEY environment variable is not set");
// }

if (process.env.OPENAI_API_KEY) {
    drivers.push({
        name: "openai",
        driver: new OpenAIDriver({
            apiKey: process.env.OPENAI_API_KEY as string
        }),
        models: [
            "gpt-4o",
        ]
    }
    )
} else {
    console.warn("OpenAI tests are skipped: OPENAI_API_KEY environment variable is not set");
}

// const AZURE_OPENAI_MODELS = [
//     "gpt-4o",
//     "gpt-3.5-turbo"
// ]


// if (process.env.AZURE_OPENAI_KEY && process.env.AZURE_OPENAI_ENDPOINT) {
//     drivers.push({
//         name: "azure-openai-key",
//         driver: new AzureOpenAIDriver({
//             apiKey: process.env.AZURE_OPENAI_KEY as string,
//             endpoint: process.env.AZURE_OPENAI_ENDPOINT as string,
//             deployment: process.env.AZURE_OPENAI_DEPLOYMENT as string
//         }),
//         models: AZURE_OPENAI_MODELS
//     })
// } else {
//     console.warn("Azure OpenAI tests are skipped: AZURE_OPENAI_KEY environment variable is not set");
// }


if (process.env.BEDROCK_REGION) {
    drivers.push({
        name: "bedrock",
        driver: new BedrockDriver({
            region: process.env.BEDROCK_REGION as string,
        }),
        //Use foundation models and inference profiles to test the driver
        models: [
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            //"us.writer.palmyra-x5-v1:0" // Only in us-west-2
        ],
    });
} else {
    console.warn("Bedrock tests are skipped: BEDROCK_REGION environment variable is not set");
}

// if (process.env.GROQ_API_KEY) {

//     drivers.push({
//         name: "groq",
//         driver: new GroqDriver({
//             apiKey: process.env.GROQ_API_KEY as string
//         }),
//         models: [
//             "llama3-70b-8192",
//             "mixtral-8x7b-32768",
//             "llama-3.3-70b-versatile"
//         ]
//     })
// } else {
//     console.warn("Groq tests are skipped: GROQ_API_KEY environment variable is not set");
// }


// if (process.env.WATSONX_API_KEY) {

//     drivers.push({
//         name: "watsonx",
//         driver: new WatsonxDriver({
//             apiKey: process.env.WATSONX_API_KEY as string,
//             projectId: process.env.WATSONX_PROJECT_ID as string,
//             endpointUrl: process.env.WATSONX_ENDPOINT_URL as string
//         }),
//         models: [
//             "ibm/granite-20b-multilingual",
//             "ibm/granite-34b-code-instruct",
//             "mistralai/mixtral-8x7b-instruct-v01"
//         ]
//     })
// } else {
//     console.warn("Watsonx tests are skipped: WATSONX_API_KEY environment variable is not set");
// }

// if (process.env.XAI_API_KEY) {

//     drivers.push({
//         name: "xai",
//         driver: new xAIDriver({
//             apiKey: process.env.XAI_API_KEY as string,
//         }),
//         models: [
//             "grok-beta",
//             "grok-vision-beta"
//         ]
//     })
// }


const PROMPT_WITH_GET_NAME_TOOL = [
    {
        role: PromptRole.user,
        content: "What is the weather in Paris?",
    },
] satisfies PromptSegment[];

function addToolResponse(prompt: PromptSegment[], tool_use_id: string, response: string): PromptSegment[] {
    return [
        ...prompt,
        {
            role: PromptRole.tool,
            tool_use_id: tool_use_id,
            content: response
        }
    ];

}

function getTestOptions(model: string): ExecutionOptions {
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
            stop_sequence: ["Haemoglobin"],
        },
        output_modality: Modalities.text,
        tools: [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    type: "object",
                    properties: {
                        "location": {
                            "type": "string",
                            "description": "The city to get the weather for, e.g. Paris, San Francisco etc."
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    };
}

describe.concurrent.each(drivers)("Driver $name", ({ name, driver, models }) => {
    test.each(models)(`${name}: generation with tools for %s`, { timeout: TIMEOUT, retry: 1 }, async (model) => {
        const options = getTestOptions(model);
        let r = await driver.execute(PROMPT_WITH_GET_NAME_TOOL, options);
        expect(r.tool_use).toBeDefined();
        expect(r.tool_use?.length).toBe(1);
        expect(r.tool_use?.[0].id).toBeDefined();
        expect(r.tool_use?.[0].tool_input).toBeDefined();
        expect(r.tool_use?.[0].tool_name).toBe("get_weather");
        const tool_use = r.tool_use!;
        r = await driver.execute([{
            role: PromptRole.tool,
            tool_use_id: tool_use[0].id,
            content: "15 degrees"
        } satisfies PromptSegment], { ...options, conversation: r.conversation });
        expect(r.result.includes("15 degrees")).toBeTruthy();
        //console.log("#######Result:", r.result, model);
    });

});