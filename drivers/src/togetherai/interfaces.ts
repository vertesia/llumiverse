interface ModelInstanceConfig {
    appearsIn: any[];
    order: number;
}

interface Config {
    stop: string[];
    prompt_format: string;
    chat_template: string;
}

interface Pricing {
    input: number;
    output: number;
    hourly: number;
}

interface Instance {
    av_zone: string;
    cluster: string;
}

interface Ask {
    [key: string]: number;
}

interface Gpu {
    [key: string]: number;
}

interface Price {
    base: number;
    fine_tune: number;
    hourly: number;
    input: number;
    output: number;
}

interface Stat {
    av_zone: string;
    cluster: string;
    capacity: number;
    qps: number;
    throughput_in: number;
    throughput_out: number;
    error_rate: number;
    retry_rate: number;
}

interface Depth {
    num_asks: number;
    num_bids: number;
    num_running: number;
    asks: Ask;
    asks_updated: string;
    gpus: Gpu;
    qps: number;
    permit_required: boolean;
    price: Price;
    throughput_in: number;
    throughput_out: number;
    stats: Stat[];
}

export interface TogetherModelInfo {
    modelInstanceConfig: ModelInstanceConfig;
    _id: string;
    name: string;
    display_name: string;
    display_type: string;
    description: string;
    license: string;
    creator_organization: string;
    hardware_label: string;
    num_parameters: number;
    show_in_playground: boolean;
    isFeaturedModel: boolean;
    context_length: number;
    config: Config;
    pricing: Pricing;
    created_at: string;
    update_at: string;
    instances: Instance[];
    access: string;
    link: string;
    descriptionLink: string;
    depth: Depth;
}

export interface TextCompletion {
    id: string;
    choices: {
        text: string,
        finish_reason: string, // stop | length ?
    }[];
    usage: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    }
    created: number;
    model: string;
    object: string;
}