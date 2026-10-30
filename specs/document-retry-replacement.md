# Atomic chunk replacement on document retry

For an existing non-ready document, successful indexing replaces its retained
chunks rather than appending a second set. Build all candidate rows and validate
the source before deleting old rows. Delete only that document's chunks in the
existing locked transaction, then insert the replacement and commit ready status,
count and revision together. A valid empty source replaces the old set with zero
chunks. Any pre-commit failure rolls back replacement and preserves prior chunks,
count and revision; the existing failure handler marks the document failed.
Ready duplicate jobs remain no-ops; other documents are unaffected. Applies to
tenant and standalone workers without changing their provider requirements.

Verify success, empty replacement, insertion failure with real PostgreSQL and
duplicate delivery. No deployment, existing-data operation, provider traffic or
generation activation. This is per-document replacement, not atomic tenant-wide
model recovery; shared retrieval compatibility checks remain in effect.
