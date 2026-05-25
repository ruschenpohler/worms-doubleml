"""
Validate raw data files against manifest.json.
Reads SHA-256 hashes from data/manifest.json and verifies
that every .dta file in data/raw/ matches its recorded digest.
Exits with code 1 on any mismatch or missing file.
"""

import hashlib
import json
import sys
from pathlib import Path


def hash_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "data" / "manifest.json"
    raw_dir = project_root / "data" / "raw"

    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    with open(manifest_path) as f:
        manifest: dict[str, str] = json.load(f)

    errors = []
    for filename, expected_hash in manifest.items():
        filepath = raw_dir / filename
        if not filepath.exists():
            errors.append(f"MISSING: {filename} not found in {raw_dir}")
            continue
        actual_hash = hash_file(filepath)
        if actual_hash != expected_hash:
            errors.append(
                f"MISMATCH: {filename}\n"
                f"  expected: {expected_hash}\n"
                f"  actual:   {actual_hash}"
            )
        else:
            print(f"OK: {filename}")

    if errors:
        print("\nValidation FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1

    print(f"\nAll {len(manifest)} files validated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
