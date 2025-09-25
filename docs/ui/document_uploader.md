# Private Document Uploader Design Iteration

This document captures the updated interaction model for the private document ingestion experience. The goal is to give each user a clear sense of what happens to their files, which embedding sets are generated, and how sharing works across public and private contexts.

## Primary user flow

1. **Landing state** – When a user opens the uploader modal they see:
   - Drag-and-drop zone with explicit "Private by default" badge.
   - Checklist summarising what will happen during ingestion (MinIO storage, dense + bitwise embeddings, submission ID tagging).
   - Quick links to organisation-wide retention policies.
2. **File selection** – Once files are added the right-hand summary pane lights up showing:
   - Total files and aggregate size.
   - Auto-generated submission ID (with option to override before upload).
   - Drop-down to choose the *sharing preference* (`Public`, `Private`, `Dual`).
   - Toggle to decide which embedding set should drive RAG for future queries.
3. **Metadata enrichment** – Users can assign:
   - Primary category (chips with autocomplete from their saved categories).
   - Additional tags (multi-select input with free-form option).
   - Optional JSON metadata editor (with schema validation and copy/paste support).
4. **Security review step** – Before upload the modal surfaces:
   - A diff-style view comparing public vs bitwise embeddings (only summary stats, not raw vectors).
   - Call-outs indicating where private embeddings will be stored (MinIO path + guardrails).
   - Confirmation checkbox to acknowledge sharing preference.
5. **Upload progress** – Progress bar per file and overall submission with two tabs:
   - *Source Upload* (streaming to MinIO).
   - *Embedding Generation* (dense + bitwise progress, with chunk count display).
6. **Completion state** – Displays:
   - Submission ID and timestamp.
   - Links to MinIO object, private embedding JSON artifact, and recent RAG activity.
   - Buttons to share public embedding, share private embedding (if allowed), or revoke access.

## Component inventory

| Component | Purpose | Notes |
|-----------|---------|-------|
| `UploaderDropZone` | Accepts file drop or click-to-upload. | Surface badge showing encryption status. |
| `SubmissionSummaryCard` | Shows submission ID, size, categories. | Allows editing submission ID before final confirmation. |
| `EmbeddingPreferenceSelector` | Radio buttons for query-time embedding scope (public vs private). | Persists preference to user profile for default retrieval. |
| `SharingPreferenceSelector` | Multi-option segmented control (`Public`, `Private`, `Dual`). | Drives the `sharing_preference` persisted with each document. |
| `MetadataForm` | Collects categories, tags, JSON metadata. | Validates JSON in-line and surfaces errors before upload. |
| `SecurityChecklist` | Highlights where assets live and what data is generated. | Links to audit log and MinIO bucket explorer. |
| `UploadProgressPanel` | Streams ingestion updates (chunk counts, MinIO URIs). | Splits progress by stage to help debug slow embedding steps. |
| `CompletionActions` | Final screen actions to share/revoke embeddings. | Exposes copy-to-clipboard for submission ID and API endpoint. |

## Responsive considerations

- On mobile the metadata form collapses into accordion sections; the security checklist becomes a swipeable carousel.
- Progress panel converts to stacked cards with inline status icons.
- Completion actions compress into a vertical button stack with icons and short labels.

## Future enhancements

- Inline preview of extracted text chunks with toggle to view bitwise representation heatmaps.
- Activity timeline that shows when public or private embeddings were shared and with whom.
- Integration with notification system to alert when someone accesses a shared embedding set.

