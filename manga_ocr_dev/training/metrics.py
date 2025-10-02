import numpy as np
import evaluate


class Metrics:
    """
    A class for computing evaluation metrics for the OCR model, specifically Character Error Rate (CER) and accuracy.
    """
    def __init__(self, processor):
        """
        Initializes the Metrics class.

        Args:
            processor: The processor used for tokenization and decoding, which is
                       essential for converting model outputs back to text.
        """
        self.cer_metric = evaluate.load("cer")
        self.processor = processor

    def compute_metrics(self, pred):
        """
        Computes the CER and accuracy for a given set of predictions and labels.

        Args:
            pred: A prediction object from the model, which contains `label_ids`
                  (the ground truth) and `predictions` (the model's output).

        Returns:
            dict: A dictionary containing the computed 'cer' and 'accuracy'.
        """
        label_ids = pred.label_ids
        pred_ids = pred.predictions
        print(label_ids.shape, pred_ids.shape)

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = np.array(["".join(text.split()) for text in pred_str])
        label_str = np.array(["".join(text.split()) for text in label_str])

        results = {}
        try:
            results["cer"] = self.cer_metric.compute(predictions=pred_str, references=label_str)
        except Exception as e:
            print(e)
            print(pred_str)
            print(label_str)
            results["cer"] = 0
        results["accuracy"] = (pred_str == label_str).mean()

        return results