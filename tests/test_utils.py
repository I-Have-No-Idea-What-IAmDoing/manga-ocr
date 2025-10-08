"""Tests for the synthetic data generator's utility functions.

This module contains unit tests for the helper functions used in the synthetic
data generation pipeline. It covers character type checking, asset loading,
and metadata processing.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from manga_ocr_dev.synthetic_data_generator.common.utils import (
    get_background_df,
    is_kanji,
    is_hiragana,
    is_katakana,
    is_ascii,
    get_charsets,
    get_font_meta,
)

def test_is_kanji():
    """Tests the `is_kanji` function for correct character identification."""
    assert is_kanji('日')
    assert not is_kanji('a')
    assert not is_kanji('あ')
    assert not is_kanji('ア')
    assert not is_kanji('1')
    assert not is_kanji('')
    assert not is_kanji('日本')

def test_is_hiragana():
    """Tests the `is_hiragana` function for correct character identification."""
    assert is_hiragana('あ')
    assert not is_hiragana('a')
    assert not is_hiragana('日')
    assert not is_hiragana('ア')
    assert not is_hiragana('1')
    assert not is_hiragana('')
    assert not is_hiragana('あいう')

def test_is_katakana():
    """Tests the `is_katakana` function for correct character identification."""
    assert is_katakana('ア')
    assert not is_katakana('a')
    assert not is_katakana('日')
    assert not is_katakana('あ')
    assert not is_katakana('1')
    assert not is_katakana('')
    assert not is_katakana('アイウ')

def test_is_ascii():
    """Tests the `is_ascii` function for correct character identification."""
    assert is_ascii('a')
    assert is_ascii('1')
    assert is_ascii('!')
    assert not is_ascii('日')
    assert not is_ascii('あ')
    assert not is_ascii('ア')
    assert not is_ascii('')
    assert not is_ascii('ab')

def test_get_background_df(tmp_path):
    """Tests the `get_background_df` function for correct parsing of filenames."""
    # Create dummy background files
    (tmp_path / "bg1_0_100_0_100.png").touch()
    (tmp_path / "bg2_50_150_50_150.png").touch()

    df = get_background_df(tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert 'path' in df.columns
    assert 'h' in df.columns
    assert 'w' in df.columns
    assert 'ratio' in df.columns
    assert df.h.iloc[0] == 100
    assert df.w.iloc[0] == 100

@patch('manga_ocr_dev.synthetic_data_generator.common.utils.pd.read_csv')
def test_get_charsets(mock_read_csv):
    """Tests the `get_charsets` function for correct character categorization."""
    mock_read_csv.return_value = pd.DataFrame({'char': ['日', 'あ', 'ア', 'a']})

    with patch('manga_ocr_dev.synthetic_data_generator.common.utils.is_hiragana', side_effect=lambda c: c == 'あ'), \
         patch('manga_ocr_dev.synthetic_data_generator.common.utils.is_katakana', side_effect=lambda c: c == 'ア'):
        vocab, hiragana, katakana = get_charsets()

    assert '日' in vocab
    assert 'あ' in hiragana
    assert 'ア' in katakana

@patch('manga_ocr_dev.synthetic_data_generator.common.utils.pd.read_csv')
def test_get_font_meta(mock_read_csv):
    """Tests the `get_font_meta` function for correct metadata loading."""
    mock_df = pd.DataFrame({
        'font_path': ['font1.ttf', 'font2.ttf'],
        'supported_chars': ['abc', 'def'],
        'label': ['regular', 'common'],
        'num_chars': [3, 3]
    })
    mock_read_csv.return_value = mock_df

    df, font_map = get_font_meta()

    assert isinstance(df, pd.DataFrame)
    assert 'font_path' in df.columns
    assert isinstance(font_map, dict)
    assert len(font_map) == 2
    assert 'font1.ttf' in df.font_path.iloc[0]
    assert 'a' in list(font_map.values())[0]