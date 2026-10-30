# Vertex AI model notes

## Palm2 for text

See https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text

### Request

`POST https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/text-bison:predict`

```json
{
  "instances": [
    {
      "prompt": string
    }
  ],
  "parameters": {
    "temperature": number,
    "maxOutputTokens": integer,
    "topK": integer,
    "topP": number,
    "groundingConfig": string,
    "stopSequences": [ string ],
    "candidateCount": integer,
    "logprobs": integer,
    "presencePenalty": float,
    "frequencyPenalty": float,
    "logitBias": map<string, float>,
    "echo": boolean,
    "seed": integer
  }
}
```

### Properties

#### prompt

Text input to generate model response. Prompts can include preamble, questions, suggestions, instructions, or examples.

### Response

```json
{
  "predictions":[
    {
      "content": string,
      "citationMetadata": {
        "citations": [
          {
            "startIndex": integer,
            "endIndex": integer,
            "url": string,
            "title": string,
            "license": string,
            "publicationDate": string
          }
        ]
      },
      "logprobs": {
        "tokenLogProbs": [ float ],
        "tokens": [ string ],
        "topLogProbs": [ { map<string, float> } ]
      },
      "safetyAttributes": {
        "categories": [ string ],
        "blocked": boolean,
        "scores": [ float ],
        "errors": [ int ]
      }
    }
  ],
  "metadata": {
    "tokenMetadata": {
      "input_token_count": {
        "total_tokens": integer,
        "total_billable_characters": integer
      },
      "output_token_count": {
        "total_tokens": integer,
        "total_billable_characters": integer
      }
    }
  }
}
```

## Code completion

See https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/code-generation

### Request:

`POST https://us-central1-googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/publishers/google/models/code-gecko:predict`

```json
{
  "instances":[
    {
      "prefix": string,
      "suffix": string
    }
  ],
  "parameters": {
    "temperature": number,
    "maxOutputTokens": integer,
    "candidateCount": integer,
    "stopSequences": [ string ],
    "logprobs": integer,
    "presencePenalty": float,
    "frequencyPenalty": float,
    "logitBias": map<string, float>,
    "echo": boolean,
    "seed": integer
  }
}
```
### Properties

#### prefix (required)

For code models, prefix represents the beginning of a piece of meaningful programming code or a natural language prompt that describes code to be generated. The model attempts to fill in the code in between the prefix and suffix.

#### suffix (optional)

For code completion, suffix represents the end of a piece of meaningful programming code. The model attempts to fill in the code in between the prefix and suffix.


### Response

```json
{
  "predictions": [
    {
      "content": string,
      "citationMetadata": {
        "citations": [
          {
            "startIndex": integer,
            "endIndex": integer,
            "url": string,
            "title": string,
            "license": string,
            "publicationDate": string
          }
        ]
      },
      "logprobs": {
        "tokenLogProbs": [ float ],
        "tokens": [ string ],
        "topLogProbs": [ { map<string, float> } ]
      },
      "safetyAttributes":{
        "categories": [ string ],
        "blocked": boolean,
        "scores": [ float ],
        "errors": [ int ]
      },
      "score": float
    }
  ]
}
```


## Code generation

See https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/code-generation

### Request

`POST https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/publishers/google/models/code-bison:predict`


```json
{
  "instances": [
    { "prefix": string }
  ],
  "parameters": {
    "temperature": number,
    "maxOutputTokens": integer,
    "candidateCount": integer,
    "stopSequences": [ string ]
    "logprobs": integer,
    "presencePenalty": float,
    "frequencyPenalty": float,
    "logitBias": map<string, float>,
    "echo": boolean,
    "seed": integer
  }
}
```
### Properties

#### prefix (required)

For code models, prefix represents the beginning of a piece of meaningful programming code or a natural language prompt that describes code to be generated. The model attempts to fill in the code in between the prefix and suffix.

### Response

```json
{
  "predictions": [
    {
      "content": string,
      "citationMetadata": {
        "citations": [
          {
            "startIndex": integer,
            "endIndex": integer,
            "url": string,
            "title": string,
            "license": string,
            "publicationDate": string
          }
        ]
      },
      "logprobs": {
        "tokenLogProbs": [ float ],
        "tokens": [ string ],
        "topLogProbs": [ { map<string, float> } ]
      },
      "safetyAttributes": {
        "categories": [ string ],
        "blocked": false,
        "scores": [ float ],
        "errors": [ int ]
      },
      "score": float
    }
  ]
}
```

## Embeddings

See https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embeddings

### Request

`POST https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/publishers/google/models/textembedding-gecko:predict`

```json
{
  "instances": [
    { "content": "TEXT"}
  ],
}
```

### Response

```json
{
  "predictions": [
    {
      "embeddings": {
        "statistics": {
          "truncated": boolean,
          "token_count": integer
        },
        "values": [ number ]
      }
    }
  ]
}
```
