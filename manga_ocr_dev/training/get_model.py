from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderConfig,
)


class TrOCRProcessorCustom:
    """
    A custom processor class that bypasses the type checks of the base class.

    This class is a wrapper around a feature extractor and a tokenizer,
    and is used to combine them into a single processor object.
    """

    def __init__(self, feature_extractor, tokenizer):
        """
        Initializes the TrOCRProcessorCustom.

        Args:
            feature_extractor: The feature extractor to use for image processing.
            tokenizer: The tokenizer to use for text processing.
        """
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor


def get_processor(encoder_name, decoder_name):
    """
    Initializes and returns a custom processor.

    Args:
        encoder_name (str): The name or path of the pre-trained encoder model.
        decoder_name (str): The name or path of the pre-trained decoder model.

    Returns:
        TrOCRProcessorCustom: A custom processor instance.
    """
    feature_extractor = AutoImageProcessor.from_pretrained(encoder_name, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    processor = TrOCRProcessorCustom(feature_extractor, tokenizer)
    return processor


def get_model(encoder_name, decoder_name, max_length, num_decoder_layers=None):
    """
    Constructs and returns a VisionEncoderDecoderModel for OCR, along with its processor.

    This function sets up the encoder and decoder from pretrained models,
    configures them for an encoder-decoder architecture, and initializes
    the VisionEncoderDecoderModel. It also sets up special tokens and
    beam search parameters for generation.

    Args:
        encoder_name (str): The name or path of the pre-trained encoder model.
        decoder_name (str): The name or path of the pre-trained decoder model.
        max_length (int): The maximum length for the generated text sequences.
        num_decoder_layers (int, optional): If specified, truncates the decoder
            to this number of layers. Defaults to None.

    Returns:
        tuple: A tuple containing the configured VisionEncoderDecoderModel and its processor.
    """
    encoder_config = AutoConfig.from_pretrained(encoder_name)
    encoder_config.is_decoder = False
    encoder_config.add_cross_attention = False
    encoder = AutoModel.from_config(encoder_config)

    decoder_config = AutoConfig.from_pretrained(decoder_name)
    decoder_config.max_length = max_length
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder = AutoModelForCausalLM.from_config(decoder_config)

    if num_decoder_layers is not None:
        if decoder_config.model_type == "bert":
            decoder.bert.encoder.layer = decoder.bert.encoder.layer[-num_decoder_layers:]
        elif decoder_config.model_type in ("roberta", "xlm-roberta"):
            decoder.roberta.encoder.layer = decoder.roberta.encoder.layer[-num_decoder_layers:]
        elif decoder_config.model_type in ("modernbert"):
            decoder.modernbert.encoder.layer = decoder.modernbert.encoder.layer[-num_decoder_layers:]
        else:
            raise ValueError(f"Unsupported model_type: {decoder_config.model_type}")

        decoder_config.num_hidden_layers = num_decoder_layers

    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    config.tie_word_embeddings = False
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

    processor = get_processor(encoder_name, decoder_name)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model, processor