import { AIModel, PromptRole, PromptSegment } from "@llumiverse/core";
import { MistralAIDriver } from "@llumiverse/drivers";
import 'dotenv/config';

async function main() {

    const model = "mistral-large-latest";
    const driver = new MistralAIDriver({
        apiKey: process.env.MISTRAL_API_KEY as string
    });

    // list models
    const models: AIModel[] = await driver.listModels();

    console.log('# Main mistral models:');
    for (const model of models) {
        console.log(`${model.name} [${model.id}]`);
    }

    // execute a model (blocking)
    const prompt: PromptSegment[] = [
        {
            role: PromptRole.user,
            content: 'Hello'
        }
    ]

    console.log(`\n# Executing model ${model} with prompt: `, prompt);
    const response = await driver.execute(prompt, {
        model,
        temperature: 0.6,
        max_tokens: 1024
    });

    console.log('\n# LLM response:', response.result)
    console.log('# Response took', response.execution_time, 'ms')
    console.log('# Token usage:', response.token_usage);

    // execute a model in streaming mode 
    console.log(`\n# Executing model ${model} in streaming mode with prompt: `, prompt);
    const stream = await driver.stream(prompt, {
        model,
        temperature: 0.6,
        max_tokens: 1024
    });

    // show the streaming response as it comes
    for await (const chunk of stream) {
        process.stdout.write(chunk);
    }

    // get the recomposed response from the stream chunks
    const streamingResponse = stream.completion!;

    console.log('\n# LLM response:', streamingResponse.result)
    console.log('# Response took', streamingResponse.execution_time, 'ms')
    console.log('# Token usage:', streamingResponse.token_usage);

    const result = await driver.execute([{ content: "Translate Hello in Romanian", role: PromptRole.user }], {
        model,
        temperature: 0.6,
        max_tokens: 1024,
        result_schema: {
            type: "object",
            property: {
                translation: {
                    type: "string"
                }
            }
        }
    });

    console.log("@@@", result.result);
    console.log("###error?", result.error);
    console.log(">>>prompt", result.prompt);
}

main().catch(console.error);
