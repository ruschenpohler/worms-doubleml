from src.cas import hash_file, store_artefact


def test_hash_file_produces_hex_digest(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello")
    digest = hash_file(f)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_store_artefact_is_idempotent(tmp_path):
    src = tmp_path / "artefact.bin"
    src.write_bytes(b"data")
    store = tmp_path / "store"
    store.mkdir()
    d1 = store_artefact(src, store)
    d2 = store_artefact(src, store)
    assert d1 == d2
    assert len(list(store.iterdir())) == 1
