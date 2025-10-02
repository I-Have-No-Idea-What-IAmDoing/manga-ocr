import numpy as np
import torch
from torchinfo import summary


def encoder_summary(model, batch_size=4):
    """
    Generates and returns a summary of the model's encoder.

    Args:
        model: The VisionEncoderDecoderModel.
        batch_size (int, optional): The batch size to use for the input size calculation.
            Defaults to 4.

    Returns:
        str: A summary of the encoder's architecture, including output size,
             number of parameters, and multiply-adds.
    """
    img_size = model.config.encoder.image_size
    return summary(
        model.encoder,
        input_size=(batch_size, 3, img_size, img_size),
        depth=3,
        col_names=["output_size", "num_params", "mult_adds"],
        device="cpu",
    )


def decoder_summary(model, batch_size=4):
    """
    Generates and returns a summary of the model's decoder.

    Args:
        model: The VisionEncoderDecoderModel.
        batch_size (int, optional): The batch size to use for creating dummy input data.
            Defaults to 4.

    Returns:
        str: A summary of the decoder's architecture, including output size,
             number of parameters, and multiply-adds.
    """
    img_size = model.config.encoder.image_size
    encoder_hidden_shape = (
        batch_size,
        (img_size // 16) ** 2 + 1,
        model.config.decoder.hidden_size,
    )
    decoder_inputs = {
        "input_ids": torch.zeros(batch_size, 1, dtype=torch.int64),
        "attention_mask": torch.ones(batch_size, 1, dtype=torch.int64),
        "encoder_hidden_states": torch.rand(encoder_hidden_shape, dtype=torch.float32),
        "return_dict": False,
    }
    return summary(
        model.decoder,
        input_data=decoder_inputs,
        depth=4,
        col_names=["output_size", "num_params", "mult_adds"],
        device="cpu",
    )


def tensor_to_image(img):
    """
    Converts a PyTorch tensor back to a displayable image format.

    The function denormalizes the tensor, clips the values to the valid
    range [0, 255], converts it to an unsigned 8-bit integer format,
    and transposes the axes to the standard image format (H, W, C).

    Args:
        img (torch.Tensor): The input image tensor, expected to be in C, H, W format
                           and normalized in the range [-1, 1].

    Returns:
        np.ndarray: The converted image as a NumPy array.
    """
    return ((img.cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)