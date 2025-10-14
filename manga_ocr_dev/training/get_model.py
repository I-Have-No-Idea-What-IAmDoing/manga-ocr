from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)
from manga_ocr_dev.training.config.schemas import ModelConfig


class TrOCRProcessorCustom:
    """A custom processor that wraps a feature extractor and a tokenizer.

    This class acts as a simple container for a feature extractor and a
    tokenizer, bypassing the type checks and complexities of the official
    `TrOCRProcessor`. It provides a streamlined interface for preparing data
    for the `VisionEncoderDecoderModel`, which requires both image and text
    processing.

    Attributes:
        feature_extractor: An image feature extractor, such as
            `AutoImageProcessor`, for converting images into tensors.
        tokenizer: A tokenizer, such as `AutoTokenizer`, for converting text
            into token IDs.
    """

    def __init__(self, feature_extractor, tokenizer):
        """Initializes the TrOCRProcessorCustom.

        Args:
            feature_extractor: The feature extractor to use for image
                processing.
            tokenizer: The tokenizer to use for text processing.
        """
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer


def get_model(model_config: ModelConfig):
    """Constructs and returns a `VisionEncoderDecoderModel` and processor for OCR.

    This function sets up a vision-encoder-decoder architecture by loading
    a pre-trained vision model as the encoder and a pre-trained language model
    as the decoder. It configures them to work together and initializes the
    `VisionEncoderDecoderModel`. The function also sets up special tokens and
    beam search parameters for effective text generation during inference.

    If `num_decoder_layers` is specified in the config, the decoder will be
    truncated to that number of layers, which can be useful for creating a
    smaller, faster model.

    Args:
        model_config: The model configuration object, specifying the encoder
            and decoder names, max length, and other parameters.

    Returns:
        A tuple containing:
            - The configured `VisionEncoderDecoderModel` for OCR.
            - The `TrOCRProcessorCustom` instance for data processing.

    Raises:
        ValueError: If `num_decoder_layers` is specified but the decoder's
            model type is not supported for layer truncation.
    """
    # Create the processor by combining a feature extractor and a tokenizer
    feature_extractor = AutoImageProcessor.from_pretrained(
        model_config.encoder_name, use_fast=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.decoder_name)
    processor = TrOCRProcessorCustom(feature_extractor, tokenizer)

    # Configure and create the encoder from the specified pre-trained model
    encoder_config = AutoConfig.from_pretrained(model_config.encoder_name)
    encoder_config.is_decoder = False
    encoder_config.add_cross_attention = False
    encoder = AutoModel.from_config(encoder_config)

    # Configure and create the decoder from the specified pre-trained model
    decoder_config = AutoConfig.from_pretrained(model_config.decoder_name)
    decoder_config.max_length = model_config.max_len
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder = AutoModelForCausalLM.from_config(decoder_config)

    # If specified, truncate the decoder to a smaller number of layers.
    # This is a form of model surgery that allows for creating a smaller,
    # faster decoder by using only the top N layers of a pretrained model.
    if model_config.num_decoder_layers is not None:
        if decoder_config.model_type == "bert":
            decoder.bert.encoder.layer = decoder.bert.encoder.layer[
                -model_config.num_decoder_layers :
            ]
        elif decoder_config.model_type in ("roberta", "xlm-roberta"):
            decoder.roberta.encoder.layer = decoder.roberta.encoder.layer[
                -model_config.num_decoder_layers :
            ]
        elif decoder_config.model_type in ("modernbert"):
            decoder.modernbert.encoder.layer = decoder.modernbert.encoder.layer[
                -model_config.num_decoder_layers :
            ]
        else:
            raise ValueError(f"Unsupported model_type for layer truncation: {decoder_config.model_type}")

        decoder_config.num_hidden_layers = model_config.num_decoder_layers

    # Combine the encoder and decoder configurations into a single VisionEncoderDecoderConfig
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )
    config.tie_word_embeddings = False
    # Create the VisionEncoderDecoderModel with the configured encoder and decoder
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

    # Set special tokens required for training and generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Configure beam search parameters for more effective inference
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = model_config.max_len
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model, processor