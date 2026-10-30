# Gemini token-limit audit

Checked against Google's model pages on 2026-10-12. Values are token counts, not execution defaults.
The window column follows Google's **Context window** or **Maximum input tokens** field; output has a separate cap.

| Model                                                                                                                     | Context/input window | Maximum output |
| ------------------------------------------------------------------------------------------------------------------------- | -------------------: | -------------: |
| [2.5 Pro](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-pro)                           |            1,048,576 |         65,536 |
| [2.5 Flash](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-flash)                       |            1,048,576 |         65,536 |
| [2.5 Flash-Lite](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-flash-lite)             |            1,048,576 |         65,536 |
| [3 Flash preview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-flash)                   |            1,048,576 |         65,536 |
| [3.1 Pro preview/customtools](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-pro)       |            1,048,576 |         65,536 |
| [3.1 Flash-Lite](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-lite)             |            1,048,576 |         65,536 |
| [3.5 Flash](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-flash)                       |            1,048,576 |         65,536 |
| [3.5 Flash-Lite](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-flash-lite)             |            1,048,576 |         65,536 |
| [3.6 Flash](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-6-flash)                       |            1,048,576 |         65,536 |
| [3.7 Flash](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-7-flash)                       |            1,048,576 |         65,536 |
| [3.8 Flash](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash)                       |            1,048,576 |         65,536 |
| [3.8 Flash Cyber](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash-cyber)           |            1,048,576 |         65,536 |
| [2.5 Flash Image](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-flash-image)           |               32,768 |         32,768 |
| [3 Pro Image](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-pro-image)                   |               65,536 |         32,768 |
| [3.1 Flash Image](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-image)           |              131,072 |         32,768 |
| [3.1 Flash-Lite Image](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-lite-image) |               65,536 |          4,096 |
| [Nano Banana 2.1](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/nano-banana-2-1)           |              131,072 |         32,768 |
| [Omni Flash preview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/omni-flash-preview)     |              131,072 |         57,920 |
| [Omni 1.1 Flash preview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/omni-1-1-flash)     |              131,072 |         57,920 |
| [3.5 Transcribe/preview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-transcribe)     |        Not published |  Not published |

Retired 2.0 Flash/Flash-Lite retain their documented 1,048,576 window and 8,192 output limits for callers that
still supply those IDs. Their archived [Flash](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-0-flash?hl=es-419)
and [Flash-Lite](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-0-flash-lite?hl=es-419)
pages document those values. The former 3 Pro preview inherits the current Pro family limits; its model page
now redirects to the model overview, so it is not counted as a separately verified current offering.

The shared family resolver supplies canonical helpers, resolved profiles, and Vertex option limits. Preview,
qualified resource IDs, and later family generations inherit the appropriate rule. This table is an audit record,
not an allowlist for routing or discovery. Live, Robotics, TTS, and embeddings remain outside the general Gemini
listing. Omni retains video-specific controls rather than gaining a chat `max_tokens` option.

Unknown Transcribe limits stay undefined. `getMaxOutputTokens` can consequently return `undefined`; consumers must
handle missing limits instead of treating a generic chat cap as verified support. No token limit is injected into
provider requests, and explicitly supplied `max_tokens` still reaches provider validation unchanged.
