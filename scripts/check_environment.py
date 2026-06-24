"""Check the local Python environment for the REM-to-Wake project.

Run from the repository root:
    python scripts/check_environment.py

This script only inspects installed packages. It does not install or change
anything.
"""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import platform
import sys


PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("sklearn", "scikit-learn"),
    ("mne", "mne"),
    ("edfio", "edfio"),
    ("pyedflib", "pyEDFlib"),
    ("wfdb", "wfdb"),
    ("jupyter", "jupyter"),
    ("notebook", "notebook"),
    ("ipykernel", "ipykernel"),
    ("torch", "torch"),
]


def get_version(distribution_name: str) -> str:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    print("Python environment")
    print("==================")
    print(f"Executable: {sys.executable}")
    print(f"Version:    {sys.version.split()[0]}")
    print(f"Platform:   {platform.platform()}")
    print()

    print("Package check")
    print("=============")
    print(f"{'Module':<14} {'Importable':<10} Version")
    print("-" * 46)

    for module_name, distribution_name in PACKAGES:
        importable = importlib.util.find_spec(module_name) is not None
        version = get_version(distribution_name)
        print(f"{module_name:<14} {str(importable):<10} {version}")

    print()
    print("No installation was performed.")


if __name__ == "__main__":
    main()
