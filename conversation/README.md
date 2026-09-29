# @llumiverse/conversation

`@llumiverse/conversation` is the experimental canonical conversation record package for Llumiverse.
It currently defines schema version `0` with the exact revision `2026-09-30.adoption.1`, also exported
as `CONVERSATION_EXPERIMENTAL_REVISION`. Version `0` has no stable compatibility promise. Documents from
earlier experimental revisions require an explicit migration and are rejected rather than guessed.

The package supports fully materialized documents. Strict Zod schemas are authoritative for turns,
content, tools, media assets, generations, accounting, receipts, context selection, compaction records,
processing configuration, lineage, diagnostics, and ingestion batches. Public TypeScript types are
inferred from those schemas. Bounded JSON preflight, document-wide structural and semantic validation,
JSON round trips, builders, type guards, basic rendering and inspection, idempotent append-only ingestion,
and deterministic Draft 2020-12 JSON Schema exports are included.

The package also defines a strict accepted-output fragment for retaining one response without its input
history, provider replay payloads, or opaque metadata. The fragment carries its response receipt,
generation and turn identities, an explicit included/omitted asset partition, and completeness markers.
It is an output-only value, not resumable conversation history. Persisted or wire values must pass
`parseAcceptedOutputFragment()` so cross-record, accounting, media, tool-call, and hydration references
are checked in addition to their Zod shape.

`Asset.content_hash`, when present, is the SHA-256 digest of the asset content bytes. Inline base64 hashes
the decoded bytes, inline text hashes well-formed UTF-8, and inline JSON hashes the UTF-8 bytes of the
sorted-key canonical JSON representation. External locators omit `content_hash` and `byte_length` unless
some other boundary has independently read and verified the bytes. Native replay hashes bind replay
payloads and are a separate contract. Experimental records created before this convention may contain
metadata or locator fingerprints in `content_hash`; the field's presence alone does not prove that those
older asset bytes were verified. Hydration boundaries must still verify resolved bytes against the hash.
Existing retained documents and receipts are not rewritten, while newly imported records and request
fingerprints can differ because their corrected integrity metadata is part of canonical request evidence.

`ConversationPreparedRequest` is the runtime durability barrier between finalized native request
preparation and provider transport. Hosts validate its complete working document with
`parseConversationPreparedRequest()`, then retain `ConversationPreparedRequestRecord` as the
privacy-safe request evidence. The full prepared document is not a bounded fragment and is not stored
when the run retention policy excludes input history. Accepted responses can be checked against the
retained record with `assertAcceptedResponseMatchesPreparedRecord()`.

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

Native execution adapters live in `@llumiverse/drivers`. OpenAI Chat Completions, OpenAI Responses,
Claude Messages, Gemini GenerateContent, and Bedrock Converse import native history, render canonical
context into provider requests, and ingest native responses into canonical records. They support
sync/stream continuation, structured JSON output, and accepted-response recovery after persistence.
Read-only legacy projections remain at compatibility boundaries. Unsupported model modalities and
incompatible protected replay fail explicitly. This adapter coverage does not imply unrestricted
model switching or adoption by every transport and model family.

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
manifests and segmented storage, general history fragments, partial working-set validation, mutation
operations, processing jobs and readiness, delivery streams, adapters for the remaining protocols, or
persisted legacy-document migrations. It does not satisfy the stable schema-version-1, core-preview,
full runtime retirement, migration, or npm publication gates.

The package remains private while its experimental publication gate is reviewed. A future publication
must freeze the intended experimental surface, verify generated-contract compatibility, and define a
release path that does not include the package in the stable Llumiverse publication set by accident.
