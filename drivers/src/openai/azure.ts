import { DefaultAzureCredential, getBearerTokenProvider } from "@azure/identity";
import { DriverOptions } from "@llumiverse/core";
import { AzureOpenAI } from "openai";
import { BaseOpenAIDriver } from "./index.js";

export interface AzureOpenAIDriverOptions extends DriverOptions {

    /**
     * The credentials to use to access Azure OpenAI
     */
    azureADTokenProvider?: any; //type with azure credentials
    
    apiKey?: string;

    endpoint?: string;

    apiVersion?: string

    deployment?: string;

}

export class AzureOpenAIDriver extends BaseOpenAIDriver {


    service: AzureOpenAI;
    provider: "azure_openai";

    constructor(opts: AzureOpenAIDriverOptions) {
        super(opts);

        if (!opts.azureADTokenProvider && !opts.apiKey) {
            opts.azureADTokenProvider = this.getDefaultAuth();
        }

        this.service = new AzureOpenAI({
            apiKey: opts.apiKey,
            azureADTokenProvider: opts.azureADTokenProvider,          
            endpoint: opts.endpoint,
            apiVersion: opts.apiVersion ?? "2024-05-01-preview",
            deployment: opts.deployment
        });
        this.provider = "azure_openai";
    }


    getDefaultAuth() {
        const scope = "https://cognitiveservices.azure.com/.default";
        const azureADTokenProvider = getBearerTokenProvider(new DefaultAzureCredential(), scope);
        return azureADTokenProvider;
    }      


}