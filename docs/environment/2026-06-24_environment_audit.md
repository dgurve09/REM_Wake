# Environment Audit

**Audit date:** 2026-06-24
**Project phase:** Block 2 setup, June 15-28
**Purpose:** Check existing Python environments before installing anything.
**Installation performed:** No

## 1. Audit Method

The reusable check script is:

```text
scripts/check_environment.py
```

It checks the active Python interpreter, platform, and whether key scientific, notebook, EDF, and machine-learning packages are importable.

## 2. Default Python

Command:

```powershell
python scripts\check_environment.py
```

Result:

| Item | Value |
|---|---|
| Python executable | default system Python |
| Python version | 3.14.3 |
| Platform | Windows-10-10.0.19045-SP0 |

Package status:

| Module | Importable | Version |
|---|---:|---|
| numpy | No | not installed |
| pandas | No | not installed |
| scipy | No | not installed |
| matplotlib | No | not installed |
| sklearn | No | not installed |
| mne | No | not installed |
| edfio | No | not installed |
| pyedflib | No | not installed |
| wfdb | No | not installed |
| jupyter | No | not installed |
| notebook | No | not installed |
| ipykernel | No | not installed |
| torch | No | not installed |

Decision: the default Python environment is not suitable for the first EDF pilot check without installing packages.

## 3. Existing Anaconda SMRI Environment

An existing Anaconda environment named `SMRI` was found and checked.

```text
<Anaconda-envs>/SMRI/python.exe
```

Command:

```powershell
& '<Anaconda-envs>\SMRI\python.exe' scripts\check_environment.py
```

Result:

| Item | Value |
|---|---|
| Python executable | existing Anaconda `SMRI` environment |
| Python version | 3.10.20 |
| Platform | Windows-10-10.0.19045-SP0 |

Package status:

| Module | Importable | Version |
|---|---:|---|
| numpy | Yes | 2.2.6 |
| pandas | Yes | 2.3.3 |
| scipy | Yes | 1.15.3 |
| matplotlib | Yes | 3.10.8 |
| sklearn | Yes | 1.7.2 |
| mne | Yes | 1.12.1 |
| edfio | Yes | 0.4.13 |
| pyedflib | No | not installed |
| wfdb | No | not installed |
| jupyter | Yes | not installed as a standalone distribution |
| notebook | No | not installed |
| ipykernel | Yes | 7.2.0 |
| torch | Yes | 2.11.0 |

Decision: the existing `SMRI` environment is suitable for the next pilot readability check because it already includes `mne` and `edfio`. No package installation is needed before attempting to read the limited `sub-53` EDF files.

## 4. Interpretation

- Use `SMRI` for the next EDF header/readability check.
- Do not install `pyedflib`, `wfdb`, or notebook packages unless the pilot check shows that they are necessary.
- The default Python 3.14 environment should not be used for this project unless a separate dependency setup decision is made.
- No raw signal data were acquired during this audit.

## 5. Next Step

Proceed to the limited `sub-53` pilot acquisition and EDF header/alignment check using the existing `SMRI` environment.
