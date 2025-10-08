import numpy as np
import evaluate
import wandb
from manga_ocr_dev.training.utils import tensor_to_image


class Metrics:
    """Computes and logs evaluation metrics for the OCR model.

    This class is designed to be used with the Hugging Face `Trainer` to
    calculate and report evaluation metrics during model training and
    validation. It computes the Character Error Rate (CER) and exact match
    accuracy. It also logs example images with their predicted and ground
    truth captions to `wandb` for qualitative analysis.

    Attributes:
        cer_metric: An instance of the CER metric from the `evaluate` library.
        processor: The processor containing the tokenizer, used for decoding
            model outputs and labels back into human-readable text.
    """

    def __init__(self, processor):
        """Initializes the Metrics class.

        Args:
            processor: The processor containing the tokenizer for decoding.
        """
        self.cer_metric = evaluate.load("cer")
        self.processor = processor

    def compute_metrics(self, pred):
        """Computes CER and accuracy for a given set of predictions.

        This method is intended to be passed to the `compute_metrics` argument
        of the `Trainer`. It takes the model's raw predictions, decodes them,
        and compares them against the ground truth labels to compute the CER
        and accuracy. It also logs a batch of evaluation samples to `wandb`.

        Args:
            pred: A prediction object from the `Trainer`, which contains
                `label_ids` (the ground truth), `predictions` (the model's
                output logits), and `inputs` (the pixel values of the images).

        Returns:
            A dictionary containing the computed 'cer' and 'accuracy' scores,
            which will be reported to `wandb` and other logging integrations.
        """
        # Extract labels, predictions, and inputs from the prediction object
        label_ids = pred.label_ids
        pred_ids = pred.predictions
        pixel_values = pred.inputs

        # Decode the predicted token IDs into strings
        pred_str = self.processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        # Replace -100 (ignore index) with the pad token ID to allow decoding of labels
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        # Decode the label token IDs into strings
        label_str = self.processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # Normalize the strings by removing all whitespace for a fair comparison
        pred_str_norm = np.array(["".join(text.split()) for text in pred_str])
        label_str_norm = np.array(["".join(text.split()) for text in label_str])

        results = {}
        try:
            # Compute the Character Error Rate (CER)
            results["cer"] = self.cer_metric.compute(
                predictions=pred_str_norm, references=label_str_norm
            )
        except Exception as e:
            # Handle cases where CER computation might fail
            print(e)
            print(pred_str_norm)
            print(label_str_norm)
            results["cer"] = 0
        # Compute the exact match accuracy
        results["accuracy"] = (pred_str_norm == label_str_norm).mean()

        # Log a batch of images with their predicted and ground truth captions to wandb
        if pixel_values is not None:
            images = [
                tensor_to_image(pixel_values[i]) for i in range(len(pixel_values))
            ]
            captions = [f"Pred: {p}\nLabel: {l}" for p, l in zip(pred_str, label_str)]
            wandb.log(
                {
                    "eval/samples": [
                        wandb.Image(img, caption=cap) for img, cap in zip(images, captions)
                    ]
                }
            )

        return results