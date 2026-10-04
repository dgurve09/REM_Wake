# Block 9 Sleep-EDF Execution Log v0.1

## 2026-10-04 metadata hash command

The first PowerShell metadata-hash command displayed seven correct hash matches but then raised `Hash mismatch` because its final Boolean condition was inverted. No file failed its hash comparison. The committed audit and independent validator replaced this shell condition and verified all seven files directly against the official annex.

## 2026-10-04 first independent validation

The initial audit correctly stored `annotation_timing=fail` and `outcome=complete_no_go`. Its generated README incorrectly summarized the timing criterion as passed, and the first validator incorrectly expected limited compatibility.

Inspection of the retained annotation summary showed:

- every annotation duration was a multiple of 30 seconds;
- annotations did not overlap;
- all scored sleep-stage intervals were within PSG support; but
- one excluded `Sleep stage ?` interval began at the end of the SC pilot PSG and extended beyond signal support.

The protocol required complete annotation support and did not permit relaxing that condition after inspection. The result narrative and validator were corrected to retain the failed timing criterion and complete no-go decision. No data, label, channel rule, or gate was changed.
