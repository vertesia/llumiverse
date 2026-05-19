

import { DriverOptions, Providers } from "@llumiverse/core";
import OpenAI from "openai";
import { BaseOpenAIDriver } from "./index.js";

export interface OpenAIDriverOptions extends DriverOptions {

    /**
     * The OpenAI api key
     */
    apiKey?: string; //type with azure credentials
    
}




export class OpenAIDriver extends BaseOpenAIDriver {

    service: OpenAI;
    readonly provider = Providers.openai;

    constructor(opts: OpenAIDriverOptions) {
        super(opts);
        this.service = new OpenAI({
            apiKey: opts.apiKey
        });
    }
}