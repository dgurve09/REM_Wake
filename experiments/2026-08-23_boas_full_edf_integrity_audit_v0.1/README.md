# BOAS Full EDF Integrity Audit v0.1

**Work date:** 2026-08-23
**Dataset:** OpenNeuro `ds005555`, snapshot `1.1.1`
**Protocol:** `docs/data/boas_full_edf_integrity_audit_plan_v0.1.md`
**Model training performed:** No

## Result

| Check | Result |
|---|---:|
| Expected EDF files | 256 |
| PSG files | 128 |
| Headband files | 128 |
| Files matching official size and SHA-256 | 256 |
| Local EDF bytes verified | 35,913,652,480 |

Every row is compared with the `SHA256E` key in the official OpenNeuro dataset mirror at tag `1.1.1`. The raw EDF files remain outside Git.

## Decision

The full local EDF acquisition matches the official annex identities.

This audit verifies file identity only. It does not replace signal alignment, signal-quality, label, split, or model-output validation.
