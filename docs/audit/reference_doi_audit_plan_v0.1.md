# Reference DOI Audit Plan v0.1

**Created:** 2026-08-23
**Scope:** Unique `doi.org` references in project Markdown files
**Purpose:** Detect nonexistent or mistyped publication and dataset identifiers

The audit extracts each unique DOI URL, requests CSL JSON through the DOI resolver, and records the returned title, type, publisher, year, and source files. A DOI passes only when the resolver returns structured metadata with a matching identifier. This checks identifier existence and basic metadata; it does not establish that every interpretation of a cited paper is correct.
