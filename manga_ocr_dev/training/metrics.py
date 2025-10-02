import numpy as np
import evaluate
import wandb
from manga_ocr_dev.training.utils import tensor_to_image


class Metrics:
    """Computes evaluation metrics for the OCR model.

    This class is responsible for calculating the Character Error Rate (CER)
    and accuracy of the model's predictions. It uses the `evaluate` library
    for the CER computation and is designed to be used with the Hugging Face
    `Trainer`.

    Attributes:
        cer_metric: An instance of the CER metric from the `evaluate` library.
        processor: The processor used for decoding model outputs and labels
            back to text.
    """

    def __init__(self, processor):
        """Initializes the Metrics class.

        Args:
            processor: The processor containing the tokenizer for decoding.
        """
        self.cer_metric = evaluate.load("cer")
        self.processor = processor

    def compute_metrics(self, pred):
        """Computes the CER and accuracy for a given set of predictions.

        This method takes the model's predictions and the ground truth labels,
        decodes them into strings, and then computes the Character Error Rate
        (CER) and the exact match accuracy.

        Args:
            pred: A prediction object from the `Trainer`, which contains
                `label_ids` (the ground truth) and `predictions` (the model's
                output logits).

        Returns:
            dict[str, float]: A dictionary containing the computed 'cer' and
            'accuracy' scores.
        """
        label_ids = pred.label_ids
        pred_ids = pred.predictions
        pixel_values = pred.inputs

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str_norm = np.array(["".join(text.split()) for text in pred_str])
        label_str_norm = np.array(["".join(text.split()) for text in label_str])

        results = {}
        try:
            results["cer"] = self.cer_metric.compute(predictions=pred_str_norm, references=label_str_norm)
        except Exception as e:
            print(e)
            print(pred_str_norm)
            print(label_str_norm)
            results["cer"] = 0
        results["accuracy"] = (pred_str_norm == label_str_norm).mean()

        # Log images to wandb
        if pixel_values is not None:
            images = [tensor_to_image(pixel_values[i]) for i in range(len(pixel_values))]
            captions = [f"Pred: {p}\nLabel: {l}" for p, l in zip(pred_str, label_str)]
            wandb.log({"eval/samples": [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]})

        return results