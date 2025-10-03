import torch
import numpy as np
from unittest.mock import patch, MagicMock

from manga_ocr_dev.training.utils import (
    encoder_summary,
    decoder_summary,
    tensor_to_image,
)


@patch('manga_ocr_dev.training.utils.summary')
def test_encoder_summary(mock_summary):
    """Tests that encoder_summary calls torchinfo.summary with correct arguments."""
    mock_model = MagicMock()
    mock_model.config.encoder.image_size = 224
    mock_model.encoder = "encoder_model"

    encoder_summary(mock_model, batch_size=2)

    mock_summary.assert_called_once()
    call_args = mock_summary.call_args[1]
    assert call_args['input_size'] == (2, 3, 224, 224)
    assert call_args['depth'] == 3
    assert call_args['device'] == 'cpu'
    assert mock_summary.call_args[0][0] == "encoder_model"


@patch('manga_ocr_dev.training.utils.summary')
def test_decoder_summary(mock_summary):
    """Tests that decoder_summary calls torchinfo.summary with correct arguments."""
    mock_model = MagicMock()
    mock_model.config.encoder.image_size = 224
    mock_model.config.decoder.hidden_size = 768
    mock_model.decoder = "decoder_model"

    decoder_summary(mock_model, batch_size=2)

    mock_summary.assert_called_once()
    call_args = mock_summary.call_args[1]
    input_data = call_args['input_data']

    assert input_data['input_ids'].shape == (2, 1)
    assert input_data['attention_mask'].shape == (2, 1)
    assert input_data['encoder_hidden_states'].shape == (2, 197, 768)
    assert call_args['depth'] == 4
    assert call_args['device'] == 'cpu'
    assert mock_summary.call_args[0][0] == "decoder_model"


def test_tensor_to_image():
    """Tests that tensor_to_image correctly converts a tensor to a NumPy image."""
    # Create a normalized tensor (C, H, W) in the range [-1, 1]
    tensor = torch.ones((3, 10, 20)) * 0.5  # Represents a gray color

    img = tensor_to_image(tensor)

    # Check output type, shape (H, W, C), and dtype
    assert isinstance(img, np.ndarray)
    assert img.shape == (10, 20, 3)
    assert img.dtype == np.uint8

    # Check that the values are correctly denormalized
    # (0.5 + 1) / 2 * 255 = 1.5 / 2 * 255 = 0.75 * 255 = 191.25 -> 191
    expected_value = int((0.5 + 1) / 2 * 255)
    assert np.all(img == expected_value)

    # Test clipping
    tensor_out_of_bounds = torch.full((3, 10, 20), 2.0)
    img_clipped = tensor_to_image(tensor_out_of_bounds)
    assert np.max(img_clipped) <= 255
    assert np.min(img_clipped) >= 0