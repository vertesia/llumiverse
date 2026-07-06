import { type AIModel, PromptRole, type PromptSegment } from '@llumiverse/core';
import { TogetherAIDriver } from '@llumiverse/drivers';
import 'dotenv/config';

async function main() {
    const model = 'mistralai/Mistral-7B-instruct-v0.1';
    const driver = new TogetherAIDriver({
        apiKey: process.env.TOGETHER_API_KEY as string,
    });

    // list models
    const models: AIModel[] = await driver.listModels();

    console.log('# Main replicate models:');
    for (const model of models) {
        console.log(`${model.name} [${model.id}]`);
    }

    // execute a model (blocking)
    const prompt: PromptSegment[] = [
        {
            role: PromptRole.user,
            content: '<INST>Hello</INST>',
        },
    ];

    console.log(`\n# Executing model ${model} with prompt: `, prompt);
    const response = await driver.execute(prompt, {
        model,
        temperature: 0.7,
        max_tokens: 256,
    });

    console.log('\n# LLM response:', response.result);
    console.log('# Response took', response.execution_time, 'ms');
    console.log('# Token usage:', response.token_usage);
    console.log('# final prompt:', response.prompt);

    // execute a model in streaming mode
    console.log(`\n# Executing model ${model} in streaming mode with prompt: `, prompt);
    const stream = await driver.stream(prompt, {
        model,
        temperature: 0.7,
        max_tokens: 1024,
    });

    // show the streaming response as it comes
    for await (const chunk of stream) {
        process.stdout.write(chunk);
    }

    // get the recomposed response from the stream chunks
    const streamingResponse = stream.completion;
    if (!streamingResponse) {
        throw new Error('Streaming completion was not populated');
    }

    console.log('\n# LLM response:', streamingResponse.result);
    console.log('# Response took', streamingResponse.execution_time, 'ms');
    console.log('# Token usage:', streamingResponse.token_usage);
}

main().catch(console.error);
