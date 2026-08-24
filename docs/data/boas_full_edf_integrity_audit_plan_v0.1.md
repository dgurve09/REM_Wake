# BOAS Full EDF Integrity Audit Plan v0.1

**Created:** 2026-08-23
**Dataset:** OpenNeuro `ds005555`, snapshot `1.1.1`
**Scope:** 256 local EDF files stored outside Git
**Model or result changes authorized:** No

## 1. Reason for the Audit

The July 4 acquisition manifest verified every EDF against the byte size returned by the OpenNeuro S3 endpoint. Byte-size agreement detects incomplete downloads but does not establish file-content identity. The `sub-53` pilot used official SHA-256 git-annex keys, but the full 256-file acquisition did not.

## 2. Fixed Method

For each expected PSG and headband EDF:

1. retrieve the small git-annex pointer from the official OpenNeuro dataset mirror at tag `1.1.1`;
2. parse the expected file size and SHA-256 from the `SHA256E` annex key;
3. compute SHA-256 for the local EDF in streaming chunks;
4. require local size and SHA-256 to match the official annex values; and
5. retain one manifest row per EDF.

The audit must cover 128 PSG files and 128 headband files. A missing pointer, missing local file, size mismatch, hash mismatch, or duplicate relative path is a failure.

## 3. Outputs

- `experiments/2026-08-23_boas_full_edf_integrity_audit_v0.1/edf_sha256_verification_v0.1.tsv`
- a concise result record in the same folder

The raw EDF files remain outside Git. The manifest records only public dataset paths, file sizes, hashes, and verification status.

## 4. Interpretation

Passing this audit establishes byte-for-byte agreement with the official snapshot represented by the annex keys. It does not validate signal quality, PSG-to-headband synchronization, labels, or model behavior; those are separate checks.
