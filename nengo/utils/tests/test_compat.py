from nengo.utils.compat import ensure_bytes


def test_ensure_bytes():
    """Test that ensure_bytes is always the right bytes value."""
    assert ensure_bytes(u"hello") == b"hello"
    assert ensure_bytes("hello") == b"hello"
