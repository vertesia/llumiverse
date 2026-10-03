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
revisioned active-context selection edits, and deterministic Draft 2020-12 JSON Schema exports are included.

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

Canonical streaming schemas describe bounded, request-scoped draft events with native positions,
request/attempt/stream identity, monotonic sequencing, terminal status, and explicit reconciliation from
draft block identities to accepted canonical records. The package includes a bounded accumulator for
validating retained event logs and final decode evidence. `@llumiverse/core` provides a finite-response
fallback and an explicit legacy string projection. Adopted OpenAI Chat Completions, OpenAI Responses,
Claude Messages, Gemini GenerateContent, and Bedrock Converse paths emit typed lifecycle events and
reconcile draft records with accepted canonical responses. This coverage depends on the driver/model
path; the contracts do not by themselves provide a durable host event log or reconnect service.

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

`planContextChange()` pins explicit active-context entry IDs to a source fingerprint and checks protected
entries, pending application calls, complete tool exchanges, and retained replay dependencies. The
caller passes its document-ordered `plan.entry_ids` to `applyContextChange()` with the expected document
and context revisions. This first slice supports `exclude` and `replace_with_compaction` with one
completed ordinary text replacement classified as `heuristic` or `semantic`. The classification records
caller-declared information loss; it does not prove semantic equivalence. New proposals cannot claim
`value_preserving`, `reversible_representation`, or `retrievable` without the corresponding validated
strategy support. Existing archived compaction records retain their full fidelity classification.
It preserves source history, generations, usage and prior
receipts; the returned change has ordered operations and a typed idempotency receipt. Disjoint
replacement ranges require an explicit first-selected placement and causal-order policy. Required
cache intent blocks an edit; automatic cache boundaries are invalidated, and a disabled cache drops
a boundary when its selected entry is removed.

```ts
import {
    applyContextChange,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    parseConversationDocument,
    planContextChange,
} from '@llumiverse/conversation';

const recordedAt = '2026-09-11T00:00:00.000Z';
const draft = createConversationDocument({ id: 'conversation-edit-example', created_at: recordedAt });
draft.turns.push(createUserTurn({
    id: 'turn-earlier',
    authority: 'ordinary',
    status: 'completed',
    timestamps: { recorded_at: recordedAt },
    model_visibility: 'include',
    provenance: { type: 'received' },
    blocks: [createTextBlock({ id: 'block-earlier', text: 'Earlier detail', format: 'plain' })],
}));
draft.context.entries.push({ id: 'entry-earlier', type: 'source_turn', turn_id: 'turn-earlier' });
const document = parseConversationDocument(draft);
const plan = await planContextChange(document, {
    expected_revision: document.revision,
    expected_context_revision: document.context.revision,
    entry_ids: ['entry-earlier'],
});
const { document: edited, change } = await applyContextChange(document, {
    operation_id: 'edit-earlier',
    expected_revision: document.revision,
    expected_context_revision: document.context.revision,
    expected_source_fingerprint: plan.source_fingerprint,
    recorded_at: recordedAt,
    entry_ids: plan.entry_ids,
    proposal: { kind: 'exclude' },
});
console.log(edited.context.entries, change.operations);
```

The library returns a new document and receipt; it does not publish them. A host must atomically
compare-and-swap the expected revision and persist the document with its receipt. Library idempotency
does not provide a database transaction or undo external tool effects.

Native execution adapters live in `@llumiverse/drivers`. OpenAI Chat Completions, OpenAI Responses,
Claude Messages, Gemini GenerateContent, and Bedrock Converse import native history, render canonical
context into provider requests, and ingest native responses into canonical records. They support
sync/stream continuation, structured JSON output, and accepted-response recovery after persistence.
The report-returning `importNativeConversationHistory()` and five protocol-specific history importers
perform pure imports without compiling a prompt or calling a provider. Each import validates and owns
its options and native history before the first asynchronous step, so subsequent caller mutation
cannot change imported records, origin metadata, tool definitions, or receipt fingerprints. Protocol selection and recorded
hosting provider evidence are explicit; the importer does not infer the host from a native wrapper or
invent a model. Results retain native call identities, media content, and supported protected replay,
and validate the canonical document before returning it. Typed failures reject unsupported shapes.
The import report separates a caller's complete/fragment/unknown declaration from continuation readiness,
which remains `not_validated`. Missing metadata/model evidence, remote media references, and missing
tool definitions remain explicit diagnostics. Import does not hydrate media, validate target compatibility,
execute processing jobs, or publish a durable imported head. Inputs and resulting documents remain
bounded; Bedrock binary media is preserved as JSON-safe base64. Native byte views use explicit
atomic byte semantics: only intrinsic byte content is protocol data. Own JavaScript annotations,
including symbol and non-enumerable properties, are excluded and never accessed; the import report
states that policy. Shadowed byte-view fields fail before conversion. Bedrock explicitly supports
standard Uint8Array and Node Buffer views, copying intrinsic bytes without enumerating their indices
or invoking caller iterators/accessors. This does not claim to preserve arbitrary JavaScript object
properties attached to a byte view. The earlier document-only Bedrock
import API remains a named compatibility boundary with its existing identity scheme; protected replay
requires supplied provider evidence.

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

Processing policy is persisted as JSON processor IDs, versions, and configuration. The core engine can
stage bounded `on_append` jobs with the accepted append, queue selected existing content, and execute
durable stages through an injected registry and exact-revision store. The append path records jobs
without invoking plugins; `runProcessingJob()` records its selected input and output before applying a
context change. `assertProcessingReady()` checks current policy, context, target, and measurement
coverage before a host prepares dependent inference. A production host must atomically commit the
returned document, dispatch and recover jobs, supply authenticated target/tokenizer measurements,
and retain pending readiness across workflow continuation. The core engine alone does not make an
accepted append ready or guarantee exactly-once external inference. Native decoded-response finalizers
now stage the response and its processing jobs through asynchronous append. Provider preparation checks
source/target-bound readiness before prepared publication and transport; imported disabled policies also
wait for unsuperseded outstanding jobs. A count-only callback cannot grant readiness. Production host
counting, post-acknowledgment scheduling and recovery remain separate integration requirements.
The exported `CONVERSATION_FOUNDATION_LIMITATIONS` lists unavailable contract areas. A bounded selected-request
archive and experimental indexed run-head/selected-text projection have separate manifest, segment and index
contracts. One-time migration of a fully materialized legacy document uses a separate finite profile: at most
32 MiB of JSON, 2 million JSON nodes, and 100,000 turns. It validates and owns the complete source before
writing records, so migration can require substantially more memory than its 32 MiB input limit. Indexed append
and selected-context reads remain bounded by the active working set after migration.
The selected archive remains explicitly unverified for execution, and the indexed text request still needs a
durably accepted prepared record and generation admission before transport. Full indexed consumer registration,
general history fragments, sparse dependency-witness validation, full processor strategy and host lifecycle
coverage, durable or reconnectable delivery streams, adapters for the remaining protocols, and authenticated
host migration and 100,000-turn benchmarks remain incomplete. This does not satisfy the stable schema-version-1,
core-preview, full runtime retirement, or npm publication gates.

The package remains private while its experimental publication gate is reviewed. A future publication
must freeze the intended experimental surface, verify generated-contract compatibility, and define a
release path that does not include the package in the stable Llumiverse publication set by accident.

Protected native replay can continue only with a recorded provider and invocation model identifier that exactly match
an explicit target. `compatibility_scope.model` names that recorded invocation route; it does not imply that a model
alias resolves to a fixed provider version. Generation `requested_model` retains the invocation identifier, while
`resolved_model` is present only when reported by the provider response (Converse reports none). No alias or model
family equivalence is inferred. Current generated Claude and Chat replay records the invocation model. Previously
persisted generated replay may use its consistent executed generation/request receipt and replay request dependency
as origin evidence. Imported history without that evidence remains inspectable with a typed unknown-origin diagnostic,
but cannot be prepared for protected continuation. Raw legacy history cannot establish origin from the next requested
model; hosts with recorded origin must explicitly import it before preparation. Named `exportLegacy*` helpers are
read-only compatibility projections and establish no readiness. A host may explicitly select a validated checkpoint
replacement; adapters do not silently discard protected state or create checkpoints.

Current pure-import operation receipts bind a versioned semantic envelope containing the exact owned native JSON
snapshot, protocol/adapter version, and validated effective import options (origin, tool schemas, completeness,
recorded time and source request identity). Omitted defaults normalize to their effective values. The stable operation
ID permits exact retries and rejects a different semantic fingerprint through the canonical append mechanism.
This operation hash is not a native-artifact-only content hash; archival hosts must track that source hash separately.
The named historical document-only Bedrock import retains its legacy operation identity and history-only hash contract.

### Materialized whole-block selectors

`resolveContextSelection` owns and resolves a revision-pinned JSON selector against edit-eligible active context, in document order.
This is an edit planner, not a generic read-only transcript query: protected entries and incomplete tool/replay cuts
are rejected even when the caller has not yet supplied an edit proposal.
Sources are all entries, explicit turn IDs, inclusive entry/turn boundaries, or disjoint ranges. Turn boundaries that
refer to more than one active split entry are rejected; explicit entry boundaries disambiguate them. Filters compose
actor kinds/IDs, tool names, top-level block IDs/types and declarative predicates over turn metadata. Filter dimensions
are AND; values within a dimension are OR. Unknown explicit turn/block IDs and overlapping ranges are errors;
predicates that match nothing return `no_match`. Dependency cuts return diagnostics and never expand the selection.

Partial edits retain unmatched ordinary source-turn blocks through deterministic context-entry references. The request
pairs the selected block map with exact original canonical context-entry references; the receipt records the map and
all deterministic remainder IDs. Whole-entry requests retain their previous shape and payload fingerprint. Exact partial
retries depend on the immutable source/replacement turns retained by every currently supported context edit. A future
logical-delete implementation must retain sufficient referenced history or introduce a receipt/revision proof before
reusing this retry contract; the current adapter does not implement deletion. Multi-block compaction replacements are
indivisible lineage units, because their format has no disjoint per-block source provenance. Selecting only some of their
blocks is rejected rather than weakening overlap validation. Text offsets, JSON Pointer and media subranges remain
unimplemented; these selectors select whole top-level blocks only.

When a new summary consumes exactly one older compaction, `supersedes_compaction_id` names it. For multiple consumed
summaries the singular marker is absent: the new source/provenance retains every selected replacement turn and the
immutable ancestor compactions remain in the document. The singular field does not describe a multi-parent lineage.

The experimental JSON preflight rejects an own `__proto__` record key (a Zod record limitation).
An entry with that string ID can be selected wholly; a partial selection needing that map key returns
`JSON_RESERVED_PROPERTY_KEY` rather than silently widening to the whole entry. Other dictionary-like identifiers
(`constructor`, `toString`) use exact own-key lookup.

### Read-only selection and slices

`resolveConversationSelection` and `sliceConversation` share the structural selector engine with the edit wrapper.
They inspect protected entries, code, pending tool exchanges and replay dependencies without mutation eligibility
checks. `sliceConversation` returns a pinned `context_view` of canonical entry/block references; it neither copies
content into another conversation nor creates a provider-ready document. Opaque replay is selectable as a whole
block and cannot be decoded through a text range or JSON Pointer. Mutation planning remains explicit and separate.

Optional `subselections` refine blocks already matched by the source and filters. They cannot introduce a filtered-out
block; matched blocks with no refinement remain whole. Half-open text ranges use Unicode code-point offsets and an
exact block fingerprint, including code text. RFC 6901 JSON Pointers use own-property lookup, canonical array indices,
and sorted-key JSON traversal order. Same-block ranges must be disjoint; output follows context/block/source order
regardless of selector input order. Requests permit at most 4096 refinements and at most 256 per block, bounding
pairwise geometry work without imposing a small media-content limit. Source and request are owned synchronously
before hashing; the result fingerprint binds the exact source document and normalized reference selection.

Typed image regions, page ranges and time ranges refine any existing source-block range. Bounds use declared retained
metadata when available; unknown extents remain inspectable. Mixed image coordinate spaces require retained dimensions.
Media evidence separates a source metadata/locator fingerprint from binary integrity: external bytes are unverified,
stored base64 bytes and Unicode text UTF-8 bytes are hashed directly; inline JSON uses explicitly labelled canonical
JSON UTF-8 bytes, and declared hash/length mismatches are explicit. Declared dimensions are not decoder verification. Unpaired-surrogate inline text cannot establish exact UTF-8 integrity and is explicitly unverified.
Read inspection always reports `mutation_ready:false`; it performs no retrieval, crop, provider I/O or readiness admission.
Subrange mutation, deletion, fork/merge and segmented working sets remain subsequent work.

### Versioned whole-block edits and changes

`planConversationEdit` and `applyConversationEdit` support contextual protect/unprotect, safe insertion at an exact
active entry boundary, and replacement of edit-eligible whole blocks. Selection is pinned to the exact document,
context revision, source fingerprint and canonical entry/block content. Inputs and source are owned before hashing.
Unselected blocks remain in order; split remainder references are deterministic and recorded in the operation.
Received inserted turns permit only ordinary completed user/program/non-generated-agent content; no executable
calls/results, generation identity, replay or elevated instruction authority can be manufactured through edits.
Contextual unprotect never changes intrinsic instruction/replay authority. Open application tool exchanges and
complete call/replay causal intervals reject interleaved insertion. Replacement provenance must bind the exact
selected source. Derived replacements remain indivisible unless precise subrange lineage is available; active
coverage follows retained ancestors so an original and its replacement cannot be reintroduced together.

Every fresh or reused append/context/edit operation returns the shared `ConversationChange` envelope. Narrow
context-change JSON remains unchanged, and retry changes use the original accepted receipt revisions, not the
current head. Existing payload fingerprints remain unchanged. New append receipts retain immutable accepted
context references and active-tool request intent (omitted versus explicitly empty are distinct). Exact redelivery
can succeed after those entries are removed or tools change, without restoring old context or tool selection.
Source turns/assets/definitions remain retained and verified. Historical receipts lacking inactive-entry evidence
or an explicit tool-selection proof fail with an evidence-unavailable error instead of guessing; unchanged-request
legacy receipts may still use retained active entries. Receipt-only references are excluded from accepted output
fragments. Future physical deletion must preserve this immutable retry evidence or supply a proved archive.

Required cache prefixes reject affected edits; auto/off preserve namespace while invalidating an affected boundary.
These APIs operate on bounded materialized documents and perform no storage publication, provider I/O or retrieval.
Precise text/JSON/media mutation lineage, active tool-definition and cache-intent edit APIs, deletion and fork/merge
remain required subsequent work. Processing change operations compose as a separate receipt family.

### Precise source-slice edits

`planConversationSliceEdit` and `applyConversationSliceEdit` accept explicit version 2 protect/replace commands.
The existing edit functions dispatch version 2 commands to the same implementation; version 1 values and payload
hashing remain unchanged. New fragments retain source revision, block hash, exact slices, transform, and fidelity in
canonical derived provenance and edit receipts. Unselected blocks remain ordered references; text complements retain
code-point order. JSON complements retain their containers (including null and empty containers), with inverse array
positions and escaped-key pointers. Authored replacements are heuristic or semantic, never a losslessness guarantee.

Partial text mutation currently accepts plain text. Markdown and code cuts require a registered format boundary
validator and are rejected without one; read-only inspection remains available. Media cuts partition an already
bounded reference using verified original inline bytes. They preserve the same asset and declare no cropping,
decoded dimensions/duration/pages, or provider projection support. Unbounded references and unverified bytes cannot
be made mutation-ready by a supplied hash.

`verifyDerivedBlockLineage` owns its input before hashing and validates retained target receipts, source block hashes,
projections, and recursive coverage. Schema parsing is structural and does not authenticate content hashes. Its
optional bounded `turn_ids` scope verifies those roots and their transitive precise lineage dependencies only. A host
may supply an authenticated, bounded selected dependency view instead of full cold history; required records cannot
be replaced with unverified hash declarations. Use the returned snapshot and only the verified roots. Provider
execution must separately run this verification at its admitted preparation boundary; this library API alone does
not activate such a host gate. Original source records and accepted receipts must remain available for exact retries;
physical deletion/segmented dependency resolution are separate retention contracts.

Processing budgets default to `measurement_policy: 'exact_only'` when that optional field is absent.
The experimental `identified_estimate` opt-in allows a named, versioned estimate profile; it does not guarantee
provider context fit or predict billed usage. Complete provider counts and exact measurements can satisfy
`exact_only`; incomplete provider counts cannot satisfy either policy. Measurement remains host-owned.
