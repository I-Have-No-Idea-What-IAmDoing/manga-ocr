import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from manga_ocr.run import (
    are_images_identical,
    get_path_key,
    process_and_write_results,
)


def test_are_images_identical():
    """
    Tests the are_images_identical function with various image comparison scenarios.
    """
    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="red")
    img3 = Image.new("RGB", (100, 100), color="blue")
    img4 = Image.new("RGB", (50, 50), color="red")
    img5_rgb = Image.new("RGB", (100, 100), color="green")
    img5_rgba = img5_rgb.convert("RGBA")

    assert are_images_identical(img1, img2)
    assert not are_images_identical(img1, img3)
    assert not are_images_identical(img1, img4)
    assert not are_images_identical(img1, None)
    assert not are_images_identical(None, img1)
    assert are_images_identical(None, None)
    assert are_images_identical(img5_rgb, img5_rgba)


def test_get_path_key(tmp_path):
    """
    Tests the get_path_key function to ensure it returns the correct key for a path.
    """
    test_file = tmp_path / "test.txt"
    test_file.touch()
    path_key = get_path_key(test_file)
    assert isinstance(path_key, tuple)
    assert path_key[0] == test_file
    assert isinstance(path_key[1], float)


@patch("pyperclip.copy")
def test_process_and_write_results_clipboard(mock_copy):
    """
    Tests process_and_write_results with clipboard output.
    """
    mocr = Mock()
    mocr.return_value = "test text"
    img = Image.new("RGB", (100, 100))

    process_and_write_results(mocr, img, "clipboard")

    mocr.assert_called_once_with(img)
    mock_copy.assert_called_once_with("test text")


def test_process_and_write_results_file(tmp_path):
    """
    Tests process_and_write_results with file output.
    """
    mocr = Mock()
    mocr.return_value = "test text"
    img = Image.new("RGB", (100, 100))
    output_file = tmp_path / "output.txt"

    process_and_write_results(mocr, img, str(output_file))

    mocr.assert_called_once_with(img)
    assert output_file.read_text(encoding="utf-8") == "test text\n"


from manga_ocr.run import run


def test_process_and_write_results_invalid_path():
    """
    Tests that process_and_write_results raises ValueError for an invalid file type.
    """
    mocr = Mock()
    img = Image.new("RGB", (100, 100))

    with pytest.raises(ValueError, match='write_to must be either "clipboard" or a path to a text file'):
        process_and_write_results(mocr, img, "output.jpg")


@patch("manga_ocr.run.MangaOcr")
@patch("PIL.ImageGrab.grabclipboard")
@patch("pyperclip.copy")
@patch("time.sleep")
def test_run_clipboard(mock_sleep, mock_pyperclip_copy, mock_grabclipboard, mock_mocr):
    """
    Tests the run function in clipboard mode.
    """
    mock_mocr_instance = mock_mocr.return_value
    mock_mocr_instance.return_value = "test ocr"

    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="blue")

    # Simulate clipboard sequence: img1, img1 (duplicate), img2, then stop
    mock_grabclipboard.side_effect = [img1, img1, img2, KeyboardInterrupt]

    with pytest.raises(KeyboardInterrupt):
        run(read_from="clipboard", write_to="clipboard")

    assert mock_grabclipboard.call_count == 4
    assert mock_mocr_instance.call_count == 2
    mock_pyperclip_copy.assert_called_with("test ocr")


@patch("manga_ocr.run.MangaOcr")
@patch("pathlib.Path.iterdir")
@patch("pyperclip.copy")
@patch("time.sleep")
def test_run_directory(mock_sleep, mock_pyperclip_copy, mock_iterdir, mock_mocr, tmp_path):
    """
    Tests the run function in directory mode.
    """
    mock_mocr_instance = mock_mocr.return_value
    mock_mocr_instance.return_value = "test ocr"

    # Create dummy image files
    img_path1 = tmp_path / "img1.png"
    img_path2 = tmp_path / "img2.png"
    Image.new("RGB", (100, 100)).save(img_path1)
    time.sleep(0.1)  # Ensure modification times are different
    Image.new("RGB", (100, 100)).save(img_path2)

    # Simulate directory scanning sequence
    mock_iterdir.side_effect = [
        [],  # Initial scan finds no images
        [img_path1],  # First scan finds one image
        [img_path1, img_path2],  # Second scan finds a new image
        KeyboardInterrupt,
    ]

    with pytest.raises(KeyboardInterrupt):
        run(read_from=str(tmp_path), write_to="clipboard")

    assert mock_iterdir.call_count == 4
    assert mock_mocr_instance.call_count == 2
    mock_pyperclip_copy.assert_called_with("test ocr")


@patch("sys.platform", "linux")
@patch("os.system", return_value=1)  # wl-copy not found
@patch("manga_ocr.run.MangaOcr")
def test_run_wayland_not_installed(mock_mocr, mock_os_system, monkeypatch):
    """
    Tests that the run function raises NotImplementedError on Wayland without wl-clipboard.
    """
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    with pytest.raises(NotImplementedError):
        run(read_from="clipboard", write_to="clipboard")






@patch("manga_ocr.run.MangaOcr")
@patch("PIL.ImageGrab.grabclipboard")
@patch("time.sleep")
@patch("loguru.logger.warning")
def test_run_clipboard_oserror_verbose(mock_logger_warning, mock_sleep, mock_grabclipboard, mock_mocr):
    """
    Tests that the run function logs a warning on OSError when verbose is True.
    """
    mock_grabclipboard.side_effect = [OSError("test error"), KeyboardInterrupt]

    with pytest.raises(KeyboardInterrupt):
        run(read_from="clipboard", write_to="clipboard", verbose=True)

    mock_logger_warning.assert_called_once_with("Error while reading from clipboard (test error)")


@patch("sys.platform", "linux")
@patch("os.system", return_value=0)  # wl-copy found
@patch("pyperclip.set_clipboard")
@patch("manga_ocr.run.MangaOcr")
@patch("PIL.ImageGrab.grabclipboard")
def test_run_wayland_installed(mock_grabclipboard, mock_mocr, mock_set_clipboard, mock_os_system, monkeypatch):
    """
    Tests that the run function sets the clipboard to wl-clipboard on Wayland.
    """
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    # Mock the rest of the function to avoid infinite loop
    with patch("manga_ocr.run.time.sleep", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            run(read_from="clipboard", write_to="clipboard")

    mock_set_clipboard.assert_called_once_with("wl-clipboard")


@patch("manga_ocr.run.MangaOcr")
@patch("PIL.ImageGrab.grabclipboard")
@patch("time.sleep")
@patch("loguru.logger.warning")
def test_run_clipboard_oserror_no_image(mock_logger_warning, mock_sleep, mock_grabclipboard, mock_mocr):
    """
    Tests that the run function handles 'cannot identify image file' error gracefully.
    """
    mock_grabclipboard.side_effect = [OSError("cannot identify image file"), KeyboardInterrupt]

    with pytest.raises(KeyboardInterrupt):
        run(read_from="clipboard", write_to="clipboard", verbose=False)

    mock_logger_warning.assert_not_called()


def test_run_raises_on_invalid_read_from_path(tmp_path):
    """
    Tests that `run` raises a ValueError if `read_from` is a file instead of a directory.
    """
    invalid_path = tmp_path / "not_a_directory.txt"
    invalid_path.touch()

    with patch("manga_ocr.run.MangaOcr"):
        with pytest.raises(ValueError, match='read_from must be either "clipboard" or a path to a directory'):
            run(read_from=str(invalid_path))


@patch("manga_ocr.run.MangaOcr")
@patch("pathlib.Path.iterdir")
@patch("time.sleep", side_effect=KeyboardInterrupt)
@patch("loguru.logger.warning")
def test_run_logs_warning_on_invalid_image_file(mock_logger_warning, mock_sleep, mock_iterdir, mock_mocr, tmp_path):
    """
    Tests that `run` logs a warning when it encounters a file that cannot be opened as an image.
    """
    invalid_file = tmp_path / "invalid_image.txt"
    invalid_file.write_text("this is not an image")

    # Simulate directory scanning: initial scan is empty, then the invalid file appears
    mock_iterdir.side_effect = [
        [],
        [invalid_file],
    ]

    with pytest.raises(KeyboardInterrupt):
        run(read_from=str(tmp_path))

    mock_logger_warning.assert_called_once()
    logged_message = mock_logger_warning.call_args[0][0]
    assert f"Error while reading file {invalid_file}" in logged_message


@patch("manga_ocr.run.MangaOcr")
@patch("PIL.ImageGrab.grabclipboard")
@patch("time.sleep")
@patch("loguru.logger.warning")
def test_run_clipboard_oserror_no_png(mock_logger_warning, mock_sleep, mock_grabclipboard, mock_mocr):
    """
    Tests that the run function handles 'target image/png not available' error gracefully.
    """
    mock_grabclipboard.side_effect = [OSError("target image/png not available"), KeyboardInterrupt]

    with pytest.raises(KeyboardInterrupt):
        run(read_from="clipboard", write_to="clipboard", verbose=False)

    mock_logger_warning.assert_not_called()