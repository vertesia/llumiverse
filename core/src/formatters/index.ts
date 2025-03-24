import { JSONSchema } from "../types.js";
import { PromptSegment } from "../types.js";

export type PromptFormatter<T = any> = (messages: PromptSegment[], schema?: JSONSchema) => T;

export * from "./commons.js";
export * from "./generic.js";
export * from "./openai.js";
export * from "./nova.js";
