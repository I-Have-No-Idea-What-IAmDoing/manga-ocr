from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderConfig,
)
from manga_ocr_dev.training.config.schemas import ModelConfig


class TrOCRProcessorCustom:
    """A custom processor that wraps a feature extractor and a tokenizer.

    This class acts as a simple container for a feature extractor and a
    tokenizer, bypassing the type checks of the base processor classes. It is
    used to prepare data for the `VisionEncoderDecoderModel`.

    Attributes:
        feature_extractor: An image feature extractor (e.g.,
            `AutoImageProcessor`).
        tokenizer: A tokenizer for processing text (e.g., `AutoTokenizer`).
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

    This function sets up an encoder and a decoder from pre-trained models,
    configures them for a vision-encoder-decoder architecture, and
    initializes the `VisionEncoderDecoderModel`. It also configures special
    tokens and beam search parameters for text generation.

    If `num_decoder_layers` is specified in the config, the decoder will be
    truncated to that number of layers.

    Args:
        model_config (ModelConfig): The model configuration object.

    Returns:
        tuple[VisionEncoderDecoderModel, TrOCRProcessorCustom]: A tuple
        containing the configured OCR model and its processor.
    """
    # Create processor
    feature_extractor = AutoImageProcessor.from_pretrained(model_config.encoder_name, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(model_config.decoder_name)
    processor = TrOCRProcessorCustom(feature_extractor, tokenizer)

    # Create model
    encoder_config = AutoConfig.from_pretrained(model_config.encoder_name)
    encoder_config.is_decoder = False
    encoder_config.add_cross_attention = False
    encoder = AutoModel.from_config(encoder_config)

    decoder_config = AutoConfig.from_pretrained(model_config.decoder_name)
    decoder_config.max_length = model_config.max_len
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder = AutoModelForCausalLM.from_config(decoder_config)

    if model_config.num_decoder_layers is not None:
        if decoder_config.model_type == "bert":
            decoder.bert.encoder.layer = decoder.bert.encoder.layer[-model_config.num_decoder_layers:]
        elif decoder_config.model_type in ("roberta", "xlm-roberta"):
            decoder.roberta.encoder.layer = decoder.roberta.encoder.layer[-model_config.num_decoder_layers:]
        elif decoder_config.model_type in ("modernbert"):
            decoder.modernbert.encoder.layer = decoder.modernbert.encoder.layer[-model_config.num_decoder_layers:]
        else:
            raise ValueError(f"Unsupported model_type: {decoder_config.model_type}")

        decoder_config.num_hidden_layers = model_config.num_decoder_layers

    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    config.tie_word_embeddings = False
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

    # Set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = model_config.max_len
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model, processor