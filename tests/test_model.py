import pytest
from app import load_local_model

def test_load_cpu_fallback():
    pipe, device = load_local_model()
    assert device == "cpu"
    assert pipe is None 