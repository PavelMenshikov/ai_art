import sys
from pathlib import Path
import pytest
import streamlit as st
from unittest.mock import patch


root = Path(__file__).parent.parent
sys.path.insert(0, str(root))


@pytest.fixture(scope="session", autouse=True)
def mock_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        yield