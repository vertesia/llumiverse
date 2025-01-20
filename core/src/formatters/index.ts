import { JSONSchema4 } from "json-schema";
import { PromptSegment } from "../types.js";

export type PromptFormatter<T = any> = (messages: PromptSegment[], schema?: JSONSchema4) => T;

export * from "./commons.js";
export * from "./generic.js";
export * from "./openai.js";
export * from "./nova.js";

