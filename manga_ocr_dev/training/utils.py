"""Utility functions for model training and inspection.

This module provides helper functions for the training pipeline, including
tools for summarizing model architectures and converting tensor data back to
a viewable image format.
"""

import numpy as np
import torch
from torchinfo import summary


def encoder_summary(model, batch_size=4):
    """Generates a summary of the model's encoder architecture.

    This function uses `torchinfo.summary` to create a textual summary of the
    encoder part of a `VisionEncoderDecoderModel`. The summary includes details
    on output size, number of parameters, and multiply-adds for each layer,
    which is useful for debugging and model inspection.

    Args:
        model (torch.nn.Module): The `VisionEncoderDecoderModel` whose encoder
            will be summarized.
        batch_size (int, optional): The batch size to use for the input shape
            in the summary. Defaults to 4.

    Returns:
        A string containing the summary of the encoder's architecture.
    """
    # Get the expected image size from the model's encoder configuration
    img_size = model.config.encoder.image_size
    # Use torchinfo.summary to generate a summary of the encoder
    return summary(
        model.encoder,
        input_size=(batch_size, 3, img_size, img_size),
        depth=3,
        col_names=["output_size", "num_params", "mult_adds"],
        device="cpu",
    )


def decoder_summary(model, batch_size=4):
    """Generates a summary of the model's decoder architecture.

    This function uses `torchinfo.summary` to create a textual summary of the
    decoder part of a `VisionEncoderDecoderModel`. It constructs dummy input
    tensors with the correct shapes to probe the decoder's architecture and
    provide details on its layers.

    Args:
        model (torch.nn.Module): The `VisionEncoderDecoderModel` whose decoder
            will be summarized.
        batch_size (int, optional): The batch size to use for creating the
            dummy input data for the summary. Defaults to 4.

    Returns:
        A string containing the summary of the decoder's architecture.
    """
    # Determine the shape of the encoder's output, which is the input to the decoder's cross-attention
    img_size = model.config.encoder.image_size
    encoder_hidden_shape = (
        batch_size,
        (img_size // 16) ** 2 + 1,
        model.config.decoder.hidden_size,
    )
    # Create a dictionary of dummy inputs that match the expected shapes of the decoder
    decoder_inputs = {
        "input_ids": torch.zeros(batch_size, 1, dtype=torch.int64),
        "attention_mask": torch.ones(batch_size, 1, dtype=torch.int64),
        "encoder_hidden_states": torch.rand(encoder_hidden_shape, dtype=torch.float32),
        "return_dict": False,
    }
    # Use torchinfo.summary to generate a summary of the decoder with the dummy inputs
    return summary(
        model.decoder,
        input_data=decoder_inputs,
        depth=4,
        col_names=["output_size", "num_params", "mult_adds"],
        device="cpu",
    )


def tensor_to_image(img):
    """Converts a PyTorch image tensor back to a displayable NumPy format.

    This function takes a PyTorch image tensor, which is typically in (C, H, W)
    format and normalized to the range [-1, 1]. It denormalizes the tensor to
    the [0, 255] range and transposes its dimensions to the standard (H, W, C)
    image format.

    Args:
        img (torch.Tensor): The input image tensor.

    Returns:
        An image as a NumPy array, suitable for display with libraries like
        Matplotlib or OpenCV.
    """
    # Denormalize the image from [-1, 1] to [0, 255], convert to uint8,
    # and then transpose the dimensions from (C, H, W) to (H, W, C) for display.
    return (
        ((img.cpu().numpy() + 1) / 2 * 255)
        .clip(0, 255)
        .astype(np.uint8)
        .transpose(1, 2, 0)
    )