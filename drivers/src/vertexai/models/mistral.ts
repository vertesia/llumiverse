import { AIModel, Completion, CompletionChunkObject, ExecutionOptions, ExecutionTokenUsage, ModelType, PromptOptions, PromptRole, PromptSegment, readStreamAsBase64 } from "@llumiverse/core";
import { asyncMap } from "@llumiverse/core/async";
import { VertexAIDriver } from "../index.js";
import { ModelDefinition } from "../models.js";

interface VertexAIMistralPrompt {

}

export class MistralModelDefinition implements ModelDefinition<VertexAIMistralPrompt> {

    model: AIModel

    constructor(modelId: string) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Text,
            can_stream: true,
        } as AIModel;
    }

    async createPrompt(_driver: VertexAIDriver, segments: PromptSegment[], options: PromptOptions): Promise<GenerateContentRequest> {
        const schema = options.result_schema;

        //Loop over segments, to process prompt
        for (const msg of segments) {

        }

        //If a schema is provided, tell the model to follow it.
        if (schema) {
            push("The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(schema));
        }


        // put system mesages first and safety last
        return {};
    }

    async requestCompletion(driver: VertexAIDriver, prompt: VertexAIMistralPrompt, options: ExecutionOptions): Promise<Completion> {
        
        
        //Use the fetch client to access the model
        driver.fetchClient;

        return {
 
        } as Completion;
    }

    async requestCompletionStream(driver: VertexAIDriver, prompt: VertexAIMistralPrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        return {} as any; //To allow it to build
        
        const model = getGenerativeModel(driver, options);
        const streamingResp = await model.generateContentStream(prompt);

        const stream = asyncMap(streamingResp.stream, async (item) => {
  
            return {
                
            };
        });

        return stream;
    }

}
