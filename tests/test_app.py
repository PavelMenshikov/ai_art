import streamlit as st
from PIL import Image
from io import BytesIO
import pytest
from unittest.mock import patch, MagicMock


@patch("streamlit.set_page_config")
@patch("streamlit.title")
@patch("streamlit.file_uploader")
@patch("streamlit.button")
def test_streamlit_mock(mock_btn, mock_up, mock_title, mock_page):  
    import app
    assert mock_page.called