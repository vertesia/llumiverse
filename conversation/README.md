# @llumiverse/conversation

`@llumiverse/conversation` is the experimental canonical conversation record package for Llumiverse.
It currently defines schema version `0` with the exact revision `2026-09-11.ingestion.1`, also exported
as `CONVERSATION_EXPERIMENTAL_REVISION`. Version `0` has no stable compatibility promise. The ingestion
revision adds strict persisted receipts and accepted-record identities; documents from the earlier
`foundation.1` revision require an explicit migration and are rejected rather than guessed.

The package supports fully materialized documents. Strict Zod schemas are authoritative for turns,
content, tools, media assets, generations, accounting, receipts, context selection, compaction records,
processing configuration, lineage, diagnostics, and ingestion batches. Public TypeScript types are
inferred from those schemas. Bounded JSON preflight, document-wide structural and semantic validation,
JSON round trips, builders, type guards, basic rendering and inspection, idempotent append-only ingestion,
and deterministic Draft 2020-12 JSON Schema exports are included.

```ts
import {
    conversationDocumentFromJson,
    conversationDocumentToJson,
    createConversationDocument,
    inspectConversation,
} from '@llumiverse/conversation';

const empty = createConversationDocument({
    id: 'conversation-1',
    created_at: '2026-09-11T00:00:00.000Z',
});

const restored = conversationDocumentFromJson(conversationDocumentToJson(empty));
console.log(inspectConversation(restored));
```

Native execution adapters live in `@llumiverse/drivers`. OpenAI Chat Completions and Claude Messages
import native history once, render canonical context into provider requests, ingest native responses
directly into canonical records, and can produce read-only legacy projections at an API compatibility
boundary. Internal retries use persisted operation receipts and stable host-supplied identities. Other
driver protocols remain on their native histories and reject canonical input so a provider switch cannot
silently discard conversation state.

```ts
import { exportLegacyConversation, OPENAI_CHAT_COMPLETIONS_PROTOCOL } from '@llumiverse/drivers';

const legacyView = exportLegacyConversation(restored, OPENAI_CHAT_COMPLETIONS_PROTOCOL);
```

Import runtime schemas from `@llumiverse/conversation/schemas` and generated schemas from
`@llumiverse/conversation/json-schema`. Ordinary `import type` use of the package is erased by
TypeScript and does not load Zod.

Always use `validateConversationDocument()`, `parseConversationDocument()`, or the JSON helpers for
untrusted input. Calling a leaf Zod schema directly performs shape validation without the bounded
preflight or document-wide reference checks. In Zod 4, recursive record parsing skips an own
`__proto__` key. The public preflight therefore rejects that key in this experimental revision rather
than accepting and rewriting it. It also rejects negative zero because JSON serialization changes it
to zero. Other valid names such as `constructor`, `toString`, and `hasOwnProperty` are preserved.
Unknown measurements stay absent; builders do not invent usage, model, timestamp, or provenance data.

Processing configurations and compaction records are inert persisted data in this revision. They do
not imply that a processor ran or that a document is ready for another request. The exported
`CONVERSATION_FOUNDATION_LIMITATIONS` lists unavailable contract areas. This revision does not implement
manifests and segmented storage, partial working-set validation, fragments, mutation operations,
processing jobs and readiness, delivery streams, adapters for the remaining protocols, or persisted
legacy-document migrations. It does not satisfy the stable schema-version-1, core-preview, full runtime
retirement, migration, or npm publication gates.

The package remains private while its experimental publication gate is reviewed. A future publication
must freeze the intended experimental surface, verify generated-contract compatibility, and define a
release path that does not include the package in the stable Llumiverse publication set by accident.
