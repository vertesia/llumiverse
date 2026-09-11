# @llumiverse/conversation

Experimental schema-first conversation records for Llumiverse. This package currently defines schema
version `0` with the exact `experimental_revision` exported as `CONVERSATION_EXPERIMENTAL_REVISION`.
Version `0` is not a stable compatibility promise.

The foundation revision supports fully materialized conversation documents. It includes strict Zod
schemas and inferred TypeScript types for turns, content, tools, media assets, generations, accounting,
receipts, context selection, compaction records, processing configuration, lineage, diagnostics, and
basic inspection. It also provides bounded JSON preflight, full-document shape and semantic validation,
JSON serialization, builders, type guards, and deterministic Draft 2020-12 JSON Schema exports.

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
`CONVERSATION_FOUNDATION_LIMITATIONS` lists unavailable contract areas. In particular, this revision
does not implement manifests and segmented storage, partial working-set validation, fragments,
mutation operations, processing jobs and readiness, delivery streams, provider adapters, or legacy
migrations. It does not satisfy the stable schema-version-1, core-preview, migration, or npm
publication gates.

The package remains private while its experimental publication gate is reviewed. A future publication
must freeze the intended experimental surface, verify generated-contract compatibility, and define a
release path that does not include the package in the stable Llumiverse publication set by accident.
