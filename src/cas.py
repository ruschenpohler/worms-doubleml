"""
Content-addressed storage (CAS) helpers.
SHA-256 hashing of directories and files for reproducibility.
"""
import hashlib
from pathlib import Path


def hash_file(path: Path) -> str:
    """Return SHA-256 hex digest of a single file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_directory(path: Path) -> str:
    """Return a stable SHA-256 digest of all files under `path`, sorted."""
    h = hashlib.sha256()
    for file in sorted(Path(path).rglob("*")):
        if file.is_file():
            h.update(str(file.relative_to(path)).encode())
            h.update(hash_file(file).encode())
    return h.hexdigest()


def store_artefact(source: Path, store_dir: Path) -> str:
    """
    Copy `source` into `store_dir` under its SHA-256 hash.
    Returns the hash. Idempotent: does not re-copy if already present.
    """
    digest = hash_file(source)
    dest = store_dir / digest
    if not dest.exists():
        import shutil
        shutil.copy2(source, dest)
    return digest
