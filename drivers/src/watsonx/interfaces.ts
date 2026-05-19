

export interface WatsonxTextGenerationPayload {
    model_id: string;
    input: string;
    parameters: {
        max_new_tokens?: number;
        time_limit?: number;
        stop_sequences?: string[];
        temperature?: number;
        top_k?: number;
        top_p?: number;
    },
    project_id: string;
}

export interface WatsonxTextGenerationResponse {
    model_id: string;
    created_at: string;
    results: {
        generated_text: string;
        generated_token_count: number;
        input_token_count: number;
        stop_reason: string;
    }[]
}

export interface GenerateEmbeddingPayload {
    model_id: string;
    inputs: string[];
    project_id: string;
}


export interface GenerateEmbeddingResponse {
    model_id: string;
    created_at: string;
    results: {
        embedding: number[];
    }[]
    input_token_count: number;
}

export interface WatsonxModelSpec {
    model_id: string;
    label: string;
    provider: string;
    source: string;
    short_description: string;
    tasks: {
        id: string;
        ratings: {
            quality: number;
        }
    }[];
    min_shot_size: number;
    tier: string;
    number_params: string;
}


export interface WatsonxListModelResponse {
    total_count: number;
    limit: number;
    resources: WatsonxModelSpec[];
}


export interface WatsonAuthToken {
    access_token: string
    refresh_token: string
    token_type: string
    expire_in: number
    expiration: number
}