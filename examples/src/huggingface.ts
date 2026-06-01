import 'dotenv/config';
import { type AIModel, PromptRole, type PromptSegment } from '@llumiverse/core';
import { HuggingFaceIEDriver } from '@llumiverse/drivers';

async function main() {
    const model = 'aws-mistral-7b-instruct-v0-1-015';
    const driver = new HuggingFaceIEDriver({
        apiKey: process.env.HUGGINGFACE_API_KEY as string,
        endpoint_url: process.env.HUGGINGFACE_ENDPOINT_URL as string,
    });

    // list models
    const models: AIModel[] = await driver.listModels();

    console.log('# Main huggingface models:');
    for (const model of models) {
        console.log(`${model.name} [${model.id}]`);
    }

    // execute a model (blocking)
    const prompt: PromptSegment[] = [
        {
            role: PromptRole.user,
            content: 'Hello',
        },
    ];

    console.log(`\n# Executing model ${model} with prompt: `, prompt);
    const response = await driver.execute(prompt, {
        model,
        temperature: 0.6,
        max_tokens: 1024,
    });

    console.log('\n# LLM response:', response.result);
    console.log('# Response took', response.execution_time, 'ms');
    console.log('# Token usage:', response.token_usage);

    // execute a model in streaming mode
    console.log(`\n# Executing model ${model} in streaming mode with prompt: `, prompt);
    const stream = await driver.stream(prompt, {
        model,
        temperature: 0.6,
        max_tokens: 1024,
    });

    // show the streaming response as it comes
    for await (const chunk of stream) {
        process.stdout.write(chunk);
    }

    // get the recomposed response from the stream chunks
    // biome-ignore lint/style/noNonNullAssertion: intentional non-null assertion; TS can't prove narrowing here
    const streamingResponse = stream.completion!;

    console.log('\n# LLM response:', streamingResponse.result);
    console.log('# Response took', streamingResponse.execution_time, 'ms');
    console.log('# Token usage:', streamingResponse.token_usage);
}

main().catch(console.error);
