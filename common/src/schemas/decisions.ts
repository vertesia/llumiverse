import { z } from 'zod';
import { ExecutionTokenUsageSchema } from './completion.js';

// Runtime schemas for decision questions and answers (`Driver.decide()`).
//
// A decision asks one or more classification or scoring questions about an input and gets back calibrated
// answers instead of generated text: a probability for a yes/no question, a probability per option for a choice,
// a probability per level for a score. Providers name their question types differently (OpenAI: predicate /
// choice / score; Gemini: noul / choice / score); llumiverse uses boolean / choice / score and drivers translate.
//
// The input is a list of `PromptSegment`s, which carry `DataSource` files, so `DecisionsOptions` is a TypeScript
// type in `types.ts` rather than a schema here; the questions and the result are plain JSON.
//
// `//` rather than `/** */` throughout: a JSDoc block immediately preceding an exported declaration is picked up
// by Vertesia's OpenAPI scanner and published as that component's `description`.

const QuestionNameSchema = z.string().min(1).meta({
    description: 'Identifies the question; its answer carries the same name. Unique within a request.',
});

const QuestionInstructionsSchema = z.string().min(1).meta({
    description: 'The question to answer about the input, in natural language.',
});

export const DecisionBooleanQuestionSchema = z
    .strictObject({
        type: z.literal('boolean'),
        name: QuestionNameSchema,
        instructions: QuestionInstructionsSchema,
    })
    .meta({
        id: 'DecisionBooleanQuestion',
        description: 'Estimate how likely a statement about the input is to be true.',
    });

export const DecisionChoiceOptionSchema = z
    .strictObject({
        value: z.string().min(1).meta({ description: 'The value returned when this option is chosen.' }),
        description: z.string().meta({ description: 'What the option means, to guide the choice.' }).optional(),
    })
    .meta({ id: 'DecisionChoiceOption' });

export const DecisionChoiceQuestionSchema = z
    .strictObject({
        type: z.literal('choice'),
        name: QuestionNameSchema,
        instructions: QuestionInstructionsSchema,
        options: z.array(DecisionChoiceOptionSchema).meta({
            description: 'The options to choose from. Providers set the allowed count (for example 2 to 255).',
        }),
    })
    .meta({ id: 'DecisionChoiceQuestion', description: 'Choose one of the supplied options.' });

export const DecisionScoreLevelSchema = z
    .strictObject({
        label: z.string().min(1).meta({ description: 'The level label, returned with the score.' }),
        description: z.string().meta({ description: 'What the level means.' }).optional(),
    })
    .meta({ id: 'DecisionScoreLevel' });

export const DecisionScoreQuestionSchema = z
    .strictObject({
        type: z.literal('score'),
        name: QuestionNameSchema,
        instructions: QuestionInstructionsSchema,
        levels: z.array(DecisionScoreLevelSchema).meta({
            description: 'The ordered levels, lowest first. Providers set the allowed count.',
        }),
    })
    .meta({ id: 'DecisionScoreQuestion', description: 'Rate the input against ordered levels.' });

export const DecisionQuestionSchema = z
    .discriminatedUnion('type', [
        DecisionBooleanQuestionSchema,
        DecisionChoiceQuestionSchema,
        DecisionScoreQuestionSchema,
    ])
    .meta({ id: 'DecisionQuestion' });

const AnswerNameSchema = z.string().meta({ description: 'The name of the question this answers.' });

const ConfidenceSchema = z.number().meta({
    description:
        "The provider's confidence in the answer, between 0 and 1. Its meaning differs by provider, so a " +
        'threshold set for one provider does not carry over to another.',
});

const ProbabilitiesSchema = z.record(z.string(), z.number());

export const DecisionBooleanAnswerSchema = z
    .strictObject({
        type: z.literal('boolean'),
        name: AnswerNameSchema,
        probability: z.number().meta({ description: 'The probability that the statement is true.' }),
    })
    .meta({ id: 'DecisionBooleanAnswer' });

export const DecisionChoiceAnswerSchema = z
    .strictObject({
        type: z.literal('choice'),
        name: AnswerNameSchema,
        value: z.string().meta({ description: 'The chosen option value.' }),
        confidence: ConfidenceSchema,
        probabilities: ProbabilitiesSchema.meta({ description: 'Probability per option value.' }),
    })
    .meta({ id: 'DecisionChoiceAnswer' });

export const DecisionScoreAnswerSchema = z
    .strictObject({
        type: z.literal('score'),
        name: AnswerNameSchema,
        score: z.number().meta({ description: 'The score as the provider reports it.' }),
        label: z.string().meta({ description: 'The label of the level the score falls on.' }).optional(),
        confidence: ConfidenceSchema,
        probabilities: ProbabilitiesSchema.meta({ description: 'Probability per level label.' }),
    })
    .meta({ id: 'DecisionScoreAnswer' });

export const DecisionRefusalAnswerSchema = z
    .strictObject({
        type: z.literal('refusal'),
        name: AnswerNameSchema,
    })
    .meta({
        id: 'DecisionRefusalAnswer',
        description: 'The model declined to answer this question. Other questions can still be answered.',
    });

export const DecisionAnswerSchema = z
    .discriminatedUnion('type', [
        DecisionBooleanAnswerSchema,
        DecisionChoiceAnswerSchema,
        DecisionScoreAnswerSchema,
        DecisionRefusalAnswerSchema,
    ])
    .meta({ id: 'DecisionAnswer' });

export const DecisionsResultSchema = z
    .strictObject({
        answers: z.array(DecisionAnswerSchema).meta({
            description: 'One answer per question, in question order, each carrying its question name.',
        }),
        model: z.string().meta({ description: 'The provider model id that answered.' }),
        token_usage: ExecutionTokenUsageSchema.meta({
            description: 'Token usage. Decisions generate no text, so `result` is the reported output count.',
        }).optional(),
    })
    .meta({ id: 'DecisionsResult' });
