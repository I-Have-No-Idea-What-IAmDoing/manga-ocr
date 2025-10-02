import numpy as np
import torch
from torchinfo import summary


def encoder_summary(model, batch_size=4):
    """Generates a summary of the model's encoder.

    This function uses `torchinfo.summary` to create a textual summary of the
    encoder part of a `VisionEncoderDecoderModel`. The summary includes
    information about output size, number of parameters, and multiply-adds for
    each layer.

    Args:
        model (VisionEncoderDecoderModel): The model whose encoder will be
            summarized.
        batch_size (int, optional): The batch size to use for the input size
            calculation in the summary. Defaults to 4.

    Returns:
        str: A string containing the summary of the encoder's architecture.
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
    """Generates a summary of the model's decoder.

    This function uses `torchinfo.summary` to create a textual summary of the
    decoder part of a `VisionEncoderDecoderModel`. It constructs dummy input
    tensors with the correct shapes to probe the decoder's architecture.

    Args:
        model (VisionEncoderDecoderModel): The model whose decoder will be
            summarized.
        batch_size (int, optional): The batch size to use for creating the
            dummy input data for the summary. Defaults to 4.

    Returns:
        str: A string containing the summary of the decoder's architecture.
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
    """Converts a PyTorch tensor back to a displayable image format.

    This function takes a PyTorch image tensor (typically in C, H, W format and
    normalized in the range [-1, 1]), denormalizes it to the [0, 255] range,
    and converts it to a NumPy array in the standard image format (H, W, C).

    Args:
        img (torch.Tensor): The input image tensor.

    Returns:
        np.ndarray: The converted image as a NumPy array, suitable for display
        with libraries like Matplotlib or OpenCV.
    """
    return ((img.cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)