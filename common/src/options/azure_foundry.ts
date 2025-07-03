import { ModelOptionsInfo, ModelOptionInfoItem, ModelOptions, OptionType, SharedOptions } from "../types.js";

// Union type of all Azure Foundry options
export type AzureFoundryOptions = AzureFoundryOpenAIOptions | AzureFoundryDeepSeekOptions | AzureFoundryThinkingOptions | AzureFoundryTextOptions | AzureFoundryImageOptions;

export interface AzureFoundryOpenAIOptions {
    _option_id: "azure-foundry-openai";
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    presence_penalty?: number;
    frequency_penalty?: number;
    stop_sequence?: string[];
    image_detail?: "low" | "high" | "auto";
    reasoning_effort?: "low" | "medium" | "high";
}

export interface AzureFoundryDeepSeekOptions {
    _option_id: "azure-foundry-deepseek";
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    stop_sequence?: string[];
}

export interface AzureFoundryThinkingOptions {
    _option_id: "azure-foundry-thinking";
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    stop_sequence?: string[];
    reasoning_effort?: "low" | "medium" | "high";
    image_detail?: "low" | "high" | "auto";
}

export interface AzureFoundryTextOptions {
    _option_id: "azure-foundry-text";
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    presence_penalty?: number;
    frequency_penalty?: number;
    stop_sequence?: string[];
    seed?: number;
}

export interface AzureFoundryImageOptions {
    _option_id: "azure-foundry-image";
    width?: number;
    height?: number;
    quality?: "standard" | "hd";
    style?: "vivid" | "natural";
    response_format?: "url" | "b64_json";
    size?: "256x256" | "512x512" | "1024x1024" | "1792x1024" | "1024x1792";
}

export function getMaxTokensLimitAzureFoundry(model: string): number | undefined {
    const modelLower = model.toLowerCase();
    
    // GPT models
    if (modelLower.includes("gpt-4o")) {
        if (modelLower.includes("mini")) {
            return 16384;
        }
        return 16384;
    }
    if (modelLower.includes("gpt-4")) {
        if (modelLower.includes("turbo")) {
            return 4096;
        }
        if (modelLower.includes("32k")) {
            return 32768;
        }
        return 8192;
    }
    if (modelLower.includes("gpt-35") || modelLower.includes("gpt-3.5")) {
        return 4096;
    }
    
    // O-series models
    if (modelLower.includes("o1")) {
        if (modelLower.includes("preview")) {
            return 32768;
        }
        if (modelLower.includes("mini")) {
            return 65536;
        }
        return 100000;
    }
    if (modelLower.includes("o3")) {
        if (modelLower.includes("mini")) {
            return 100000;
        }
        return 100000;
    }
    if (modelLower.includes("o4")) {
        return 100000;
    }
    
    // DeepSeek models
    if (modelLower.includes("deepseek")) {
        if (modelLower.includes("r1")) {
            return 163840;
        }
        if (modelLower.includes("v3")) {
            return 131072;
        }
    }
    
    // Claude models
    if (modelLower.includes("claude")) {
        if (modelLower.includes("3-5") || modelLower.includes("3-7")) {
            return 8192;
        }
        if (modelLower.includes("3")) {
            return 4096;
        }
        return 4096;
    }
    
    // Llama models
    if (modelLower.includes("llama")) {
        if (modelLower.includes("3.1") || modelLower.includes("3.3")) {
            return 8192;
        }
        if (modelLower.includes("4")) {
            return 1000000; // 1M context
        }
        return 8192;
    }
    
    // Mistral models
    if (modelLower.includes("mistral")) {
        if (modelLower.includes("large")) {
            return 4096;
        }
        if (modelLower.includes("small")) {
            return 4096;
        }
        return 4096;
    }
    
    // Phi models
    if (modelLower.includes("phi")) {
        return 4096;
    }
    
    // AI21 Jamba models
    if (modelLower.includes("jamba")) {
        return 4096;
    }
    
    // Cohere models
    if (modelLower.includes("cohere")) {
        if (modelLower.includes("command-a")) {
            return 8000;
        }
        return 4096;
    }
    
    // Grok models
    if (modelLower.includes("grok")) {
        return 131072;
    }
    
    return undefined;
}

export function getAzureFoundryOptions(model: string, _option?: ModelOptions): ModelOptionsInfo {
    const modelLower = model.toLowerCase();
    const max_tokens_limit = getMaxTokensLimitAzureFoundry(model);
    
    // Image generation models
    if (modelLower.includes("dall-e") || modelLower.includes("gpt-image")) {
        return {
            _option_id: "azure-foundry-image",
            options: [
                {
                    name: "size",
                    type: OptionType.enum,
                    enum: {
                        "256x256": "256x256",
                        "512x512": "512x512", 
                        "1024x1024": "1024x1024",
                        "1792x1024": "1792x1024",
                        "1024x1792": "1024x1792"
                    },
                    default: "1024x1024",
                    description: "The size of the generated image"
                },
                {
                    name: "quality",
                    type: OptionType.enum,
                    enum: { "Standard": "standard", "HD": "hd" },
                    default: "standard",
                    description: "The quality of the generated image"
                },
                {
                    name: "style",
                    type: OptionType.enum,
                    enum: { "Vivid": "vivid", "Natural": "natural" },
                    default: "vivid",
                    description: "The style of the generated image"
                },
                {
                    name: "response_format",
                    type: OptionType.enum,
                    enum: { "URL": "url", "Base64 JSON": "b64_json" },
                    default: "url",
                    description: "The format of the response"
                }
            ]
        };
    }
    
    // Vision model options
    const visionOptions: ModelOptionInfoItem[] = isVisionModel(modelLower) ? [
        {
            name: "image_detail",
            type: OptionType.enum,
            enum: { "Low": "low", "High": "high", "Auto": "auto" },
            default: "auto",
            description: "Controls how the model processes input images"
        }
    ] : [];
    
    // O-series and thinking models
    if (modelLower.includes("o1") || modelLower.includes("o3") || modelLower.includes("o4")) {
        const reasoningOptions: ModelOptionInfoItem[] = (modelLower.includes("o3") || isO1Full(modelLower)) ? [
            {
                name: "reasoning_effort",
                type: OptionType.enum,
                enum: { "Low": "low", "Medium": "medium", "High": "high" },
                default: "medium",
                description: "How much effort the model should put into reasoning"
            }
        ] : [];
        
        return {
            _option_id: "azure-foundry-thinking",
            options: [
                {
                    name: SharedOptions.max_tokens,
                    type: OptionType.numeric,
                    min: 1,
                    max: max_tokens_limit,
                    integer: true,
                    description: "The maximum number of tokens to generate"
                },
                {
                    name: SharedOptions.temperature,
                    type: OptionType.numeric,
                    min: 0.0,
                    max: 2.0,
                    default: 1.0,
                    step: 0.1,
                    description: "Controls randomness in the output"
                },
                {
                    name: SharedOptions.top_p,
                    type: OptionType.numeric,
                    min: 0,
                    max: 1,
                    step: 0.1,
                    description: "Nucleus sampling parameter"
                },
                {
                    name: SharedOptions.stop_sequence,
                    type: OptionType.string_list,
                    value: [],
                    description: "Sequences where the model will stop generating"
                },
                ...reasoningOptions,
                ...visionOptions
            ]
        };
    }
    
    // DeepSeek R1 models
    if (modelLower.includes("deepseek") && modelLower.includes("r1")) {
        return {
            _option_id: "azure-foundry-deepseek",
            options: [
                {
                    name: SharedOptions.max_tokens,
                    type: OptionType.numeric,
                    min: 1,
                    max: max_tokens_limit,
                    integer: true,
                    description: "The maximum number of tokens to generate"
                },
                {
                    name: SharedOptions.temperature,
                    type: OptionType.numeric,
                    min: 0.0,
                    max: 2.0,
                    default: 0.7,
                    step: 0.1,
                    description: "Lower temperatures recommended for DeepSeek R1 (0.3-0.7)"
                },
                {
                    name: SharedOptions.top_p,
                    type: OptionType.numeric,
                    min: 0,
                    max: 1,
                    step: 0.1,
                    description: "Nucleus sampling parameter"
                },
                {
                    name: SharedOptions.stop_sequence,
                    type: OptionType.string_list,
                    value: [],
                    description: "Sequences where the model will stop generating"
                }
            ]
        };
    }
    
    // OpenAI models (GPT-4, GPT-4o, GPT-3.5)
    if (modelLower.includes("gpt-")) {
        return {
            _option_id: "azure-foundry-openai",
            options: [
                {
                    name: SharedOptions.max_tokens,
                    type: OptionType.numeric,
                    min: 1,
                    max: max_tokens_limit,
                    integer: true,
                    step: 200,
                    description: "The maximum number of tokens to generate"
                },
                {
                    name: SharedOptions.temperature,
                    type: OptionType.numeric,
                    min: 0.0,
                    max: 2.0,
                    default: 0.7,
                    step: 0.1,
                    description: "Controls randomness in the output"
                },
                {
                    name: SharedOptions.top_p,
                    type: OptionType.numeric,
                    min: 0,
                    max: 1,
                    step: 0.1,
                    description: "Nucleus sampling parameter"
                },
                {
                    name: SharedOptions.presence_penalty,
                    type: OptionType.numeric,
                    min: -2.0,
                    max: 2.0,
                    step: 0.1,
                    description: "Penalize new tokens based on their presence in the text"
                },
                {
                    name: SharedOptions.frequency_penalty,
                    type: OptionType.numeric,
                    min: -2.0,
                    max: 2.0,
                    step: 0.1,
                    description: "Penalize new tokens based on their frequency in the text"
                },
                {
                    name: SharedOptions.stop_sequence,
                    type: OptionType.string_list,
                    value: [],
                    description: "Sequences where the model will stop generating"
                },
                ...visionOptions
            ]
        };
    }
    
    // General text models (Claude, Llama, Mistral, Phi, etc.)
    const baseOptions: ModelOptionInfoItem[] = [
        {
            name: SharedOptions.max_tokens,
            type: OptionType.numeric,
            min: 1,
            max: max_tokens_limit,
            integer: true,
            step: 200,
            description: "The maximum number of tokens to generate"
        },
        {
            name: SharedOptions.temperature,
            type: OptionType.numeric,
            min: 0.0,
            max: 2.0,
            default: 0.7,
            step: 0.1,
            description: "Controls randomness in the output"
        },
        {
            name: SharedOptions.top_p,
            type: OptionType.numeric,
            min: 0,
            max: 1,
            step: 0.1,
            description: "Nucleus sampling parameter"
        },
        {
            name: SharedOptions.stop_sequence,
            type: OptionType.string_list,
            value: [],
            description: "Sequences where the model will stop generating"
        }
    ];
    
    // Add model-specific options
    const additionalOptions: ModelOptionInfoItem[] = [];
    
    // Add top_k for certain models
    if (modelLower.includes("claude") || modelLower.includes("mistral") || modelLower.includes("phi")) {
        additionalOptions.push({
            name: SharedOptions.top_k,
            type: OptionType.numeric,
            min: 1,
            integer: true,
            step: 1,
            description: "Limits token sampling to the top k tokens"
        });
    }
    
    // Add penalty options for certain models
    if (modelLower.includes("claude") || modelLower.includes("jamba") || modelLower.includes("cohere")) {
        additionalOptions.push(
            {
                name: SharedOptions.presence_penalty,
                type: OptionType.numeric,
                min: -2.0,
                max: 2.0,
                step: 0.1,
                description: "Penalize new tokens based on their presence in the text"
            },
            {
                name: SharedOptions.frequency_penalty,
                type: OptionType.numeric,
                min: -2.0,
                max: 2.0,
                step: 0.1,
                description: "Penalize new tokens based on their frequency in the text"
            }
        );
    }
    
    // Add seed option for certain models
    if (modelLower.includes("mistral") || modelLower.includes("phi") || modelLower.includes("gemini")) {
        additionalOptions.push({
            name: SharedOptions.seed,
            type: OptionType.numeric,
            integer: true,
            description: "Random seed for reproducible generation"
        });
    }
    
    return {
        _option_id: "azure-foundry-text",
        options: [
            ...baseOptions,
            ...additionalOptions,
            ...visionOptions
        ]
    };
}

function isVisionModel(modelLower: string): boolean {
    return modelLower.includes("gpt-4o") || 
           modelLower.includes("gpt-4-turbo") || 
           modelLower.includes("claude-3") || 
           modelLower.includes("llama-3.2") ||
           modelLower.includes("llama-4") ||
           modelLower.includes("gemini") ||
           isO1Full(modelLower);
}

function isO1Full(modelLower: string): boolean {
    if (modelLower.includes("o1")) {
        return !modelLower.includes("mini") && !modelLower.includes("preview");
    }
    return false;
}