import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
from manga_ocr_dev.training.train import main

class TestTrain(unittest.TestCase):

    @patch('manga_ocr_dev.training.train.Seq2SeqTrainer')
    @patch('manga_ocr_dev.training.train.Seq2SeqTrainingArguments')
    @patch('manga_ocr_dev.training.train.Metrics')
    @patch('manga_ocr_dev.training.train.MangaDataset')
    @patch('manga_ocr_dev.training.train.torch.compile')
    @patch('manga_ocr_dev.training.train.get_model')
    @patch('manga_ocr_dev.training.train.wandb')
    @patch('manga_ocr_dev.training.train.load_config')
    @patch('manga_ocr_dev.training.train.TRAIN_ROOT', Path('train'))
    def test_main_default_run(self, mock_load_config, mock_wandb, MockGetModel, mock_torch_compile, MockMangaDataset, MockMetrics, MockSeq2SeqTrainingArguments, MockSeq2SeqTrainer):
        """Test the main training function with a default run name."""
        # --- Setup Mocks ---
        # Mock config
        mock_config = MagicMock()
        mock_config.training.torch_compile = False
        mock_config.model.max_len = 300
        mock_config.training.model_dump.return_value = {}
        mock_load_config.return_value = mock_config

        # Mock wandb
        mock_wandb.run.name = "generated-run-name"

        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        MockGetModel.return_value = (mock_model, mock_processor)

        # Mock trainer
        mock_trainer = MagicMock()
        MockSeq2SeqTrainer.return_value = mock_trainer

        # --- Call the function ---
        main(config_path=None, run_name=None)

        # --- Assertions ---
        mock_load_config.assert_called_once_with(None)
        mock_wandb.init.assert_called_once_with(project="manga-ocr", name=None, config=ANY)
        MockGetModel.assert_called_once_with(mock_config.model)
        mock_torch_compile.assert_not_called()
        self.assertEqual(MockMangaDataset.call_count, 2)
        MockMetrics.assert_called_once_with(mock_processor)
        MockSeq2SeqTrainingArguments.assert_called_once_with(
            output_dir=str(Path('train') / "generated-run-name"),
            run_name="generated-run-name",
            **{}
        )
        MockSeq2SeqTrainer.assert_called_once()
        mock_trainer.train.assert_called_once()

    @patch('manga_ocr_dev.training.train.Seq2SeqTrainer')
    @patch('manga_ocr_dev.training.train.Seq2SeqTrainingArguments')
    @patch('manga_ocr_dev.training.train.Metrics')
    @patch('manga_ocr_dev.training.train.MangaDataset')
    @patch('manga_ocr_dev.training.train.torch.compile')
    @patch('manga_ocr_dev.training.train.get_model')
    @patch('manga_ocr_dev.training.train.wandb')
    @patch('manga_ocr_dev.training.train.load_config')
    @patch('manga_ocr_dev.training.train.TRAIN_ROOT', Path('train'))
    def test_main_with_custom_run_name_and_compile(self, mock_load_config, mock_wandb, MockGetModel, mock_torch_compile, MockMangaDataset, MockMetrics, MockSeq2SeqTrainingArguments, MockSeq2SeqTrainer):
        """Test the main training function with a custom run name and torch.compile enabled."""
        # --- Setup Mocks ---
        # Mock config
        mock_config = MagicMock()
        mock_config.training.torch_compile = True # Enable torch.compile
        mock_config.model.max_len = 300
        mock_config.training.model_dump.return_value = {'batch_size': 32}
        mock_load_config.return_value = mock_config

        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        MockGetModel.return_value = (mock_model, mock_processor)

        # --- Call the function ---
        custom_run_name = "my-custom-run"
        main(config_path=Path("/fake/path"), run_name=custom_run_name)

        # --- Assertions ---
        mock_load_config.assert_called_once_with(Path("/fake/path"))
        mock_wandb.init.assert_called_once_with(project="manga-ocr", name=custom_run_name, config=ANY)
        mock_torch_compile.assert_called_once_with(mock_model)
        MockSeq2SeqTrainingArguments.assert_called_once_with(
            output_dir=str(Path('train') / custom_run_name),
            run_name=custom_run_name,
            **{'batch_size': 32}
        )
        MockSeq2SeqTrainer.return_value.train.assert_called_once()