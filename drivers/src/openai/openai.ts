

import { DriverOptions } from "@llumiverse/core";
import OpenAI from "openai";
import { BaseOpenAIDriver } from "./index.js";

export interface OpenAIDriverOptions extends DriverOptions {

    /**
     * The OpenAI api key
     */
    apiKey?: string; //type with azure credntials
    
}




export class OpenAIDriver extends BaseOpenAIDriver {

    service: OpenAI;
    provider: "openai";

    constructor(opts: OpenAIDriverOptions) {
        super(opts);
        this.service = new OpenAI({
            apiKey: opts.apiKey
        });
        this.provider = "openai";
    }


}