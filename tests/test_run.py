"""Tests for the command-line interface and background monitoring.

This module contains tests for the functions in `manga_ocr.run`, which
implements the clipboard and directory monitoring logic for the `manga_ocr`
command-line tool. The tests cover image comparison, result writing, and the
main `run` loop's behavior under various conditions.
"""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from manga_ocr.run import (
    are_images_identical,
    get_path_key,
    process_and_write_results,
    run,
)


def test_are_images_identical():
    """Tests the `are_images_identical` function with various scenarios.

    This test covers a comprehensive set of cases to ensure the image
    comparison logic is robust. It checks:
    - Identical images.
    - Images with different colors or sizes.
    - Images with different but compatible color modes (e.g., RGB vs. RGBA).
    - Comparisons involving `None`.
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
    """Tests that `get_path_key` returns a correct and unique key.

    This test verifies that `get_path_key` generates a key for a file path
    that is a tuple containing the `Path` object and its last modification
    time. This key is essential for the directory monitoring mode to uniquely
    identify files and detect changes.

    Args:
        tmp_path: The pytest fixture for a temporary directory.
    """
    test_file = tmp_path / "test.txt"
    test_file.touch()
    path_key = get_path_key(test_file)
    assert isinstance(path_key, tuple)
    assert path_key[0] == test_file
    assert isinstance(path_key[1], float)


@patch("pyperclip.copy")
def test_process_and_write_results_to_clipboard(mock_copy):
    """Tests writing OCR results to the clipboard.

    This test verifies that the `process_and_write_results` function
    correctly calls the OCR model, gets the text result, and then uses
    `pyperclip.copy` to write the result to the system clipboard when
    `write_to` is set to "clipboard".

    Args:
        mock_copy: A mock for the `pyperclip.copy` function.
    """
    mocr = Mock()
    mocr.return_value = "test text"
    img = Image.new("RGB", (100, 100))

    process_and_write_results(mocr, img, "clipboard")

    mocr.assert_called_once_with(img)
    mock_copy.assert_called_once_with("test text")


def test_process_and_write_results_to_file(tmp_path):
    """Tests writing OCR results to a file.

    This test verifies that the `process_and_write_results` function
    correctly appends the recognized text to a specified file. It ensures
    that the file is created with the correct content and that a newline
    character is added.

    Args:
        tmp_path: The pytest fixture for a temporary directory.
    """
    mocr = Mock()
    mocr.return_value = "test text"
    img = Image.new("RGB", (100, 100))
    output_file = tmp_path / "output.txt"

    process_and_write_results(mocr, img, str(output_file))

    mocr.assert_called_once_with(img)
    assert output_file.read_text(encoding="utf-8") == "test text\n"


def test_process_and_write_results_raises_for_invalid_path():
    """Tests that `process_and_write_results` raises an error for an invalid path.

    This test ensures that the function validates the `write_to` argument and
    raises a `ValueError` if the specified file path does not have a `.txt`
    extension. This prevents the function from writing to unintended file types.
    """
    mocr = Mock()
    img = Image.new("RGB", (100, 100))

    with pytest.raises(ValueError, match='write_to must be either "clipboard" or a path to a text file'):
        process_and_write_results(mocr, img, "output.jpg")


@patch("manga_ocr.run.MangaOcr")
@patch("PIL.ImageGrab.grabclipboard")
@patch("pyperclip.copy")
@patch("time.sleep")
def test_run_in_clipboard_mode(mock_sleep, mock_pyperclip_copy, mock_grabclipboard, mock_mocr):
    """Tests the main `run` function's behavior in clipboard monitoring mode.

    This test simulates a sequence of clipboard events to verify that the OCR
    is triggered only for new, unique images. The sequence includes a new
    image, a duplicate of that image, and then another new image. The test
    asserts that the OCR function is called only twice.

    Args:
        mock_sleep: Mock for `time.sleep` to prevent delays.
        mock_pyperclip_copy: Mock for `pyperclip.copy`.
        mock_grabclipboard: Mock for `ImageGrab.grabclipboard` to simulate
            clipboard content.
        mock_mocr: Mock for the `MangaOcr` class.
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
    # OCR should be called for img1 and img2, but not the duplicate img1
    assert mock_mocr_instance.call_count == 2
    mock_pyperclip_copy.assert_called_with("test ocr")


@patch("manga_ocr.run.MangaOcr")
@patch("pathlib.Path.iterdir")
@patch("pyperclip.copy")
@patch("time.sleep")
def test_run_in_directory_mode(mock_sleep, mock_pyperclip_copy, mock_iterdir, mock_mocr, tmp_path):
    """Tests the main `run` function's behavior in directory monitoring mode.

    This test simulates the discovery of new image files in a directory over
    several polling cycles. It verifies that the OCR is triggered correctly
    for each new file and that existing files are not re-processed.

    Args:
        mock_sleep: Mock for `time.sleep` to prevent delays.
        mock_pyperclip_copy: Mock for `pyperclip.copy`.
        mock_iterdir: Mock for `pathlib.Path.iterdir` to simulate file
            discovery.
        mock_mocr: Mock for the `MangaOcr` class.
        tmp_path: The pytest fixture for a temporary directory.
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
def test_run_raises_on_wayland_without_wl_clipboard(mock_mocr, mock_os_system, monkeypatch):
    """Tests that `run` raises an error on Wayland without wl-clipboard.

    This test simulates running the application on a Linux system with Wayland
    but without the `wl-clipboard` package installed. It verifies that the
    application correctly detects this situation and raises a
    `NotImplementedError` to inform the user about the missing dependency.

    Args:
        mock_mocr: Mock for the `MangaOcr` class.
        mock_os_system: Mock for `os.system` to simulate the absence of
            `wl-copy`.
        monkeypatch: The pytest fixture for modifying environment variables.
    """
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    with pytest.raises(NotImplementedError):
        run(read_from="clipboard", write_to="clipboard")


@patch("manga_ocr.run.MangaOcr")
@patch("PIL.ImageGrab.grabclipboard")
@patch("time.sleep")
@patch("loguru.logger.warning")
def test_run_logs_clipboard_oserror_when_verbose(mock_logger_warning, mock_sleep, mock_grabclipboard, mock_mocr):
    """Tests that `run` logs a warning on OSError when `verbose=True`.

    This test ensures that when an `OSError` occurs while reading from the
    clipboard, and the `verbose` flag is enabled, the error is logged as a
    warning. This is important for providing users with feedback on potential
    issues during execution.

    Args:
        mock_logger_warning: Mock for the logger's warning method.
        mock_sleep: Mock for `time.sleep` to prevent delays.
        mock_grabclipboard: Mock for `ImageGrab.grabclipboard` to simulate
            an error.
        mock_mocr: Mock for the `MangaOcr` class.
    """
    mock_grabclipboard.side_effect = [OSError("test error"), KeyboardInterrupt]

    with pytest.raises(KeyboardInterrupt):
        run(read_from="clipboard", write_to="clipboard", verbose=True)

    mock_logger_warning.assert_called_once_with("Error while reading from clipboard: test error")


@patch("sys.platform", "linux")
@patch("os.system", return_value=0)  # wl-copy found
@patch("pyperclip.set_clipboard")
@patch("manga_ocr.run.MangaOcr")
@patch("PIL.ImageGrab.grabclipboard")
def test_run_sets_up_wayland_clipboard(mock_grabclipboard, mock_mocr, mock_set_clipboard, mock_os_system, monkeypatch):
    """Tests that `run` correctly configures `pyperclip` for Wayland.

    This test simulates an environment where the application is running on
    Linux with Wayland and `wl-clipboard` is installed. It verifies that the
    `run` function correctly detects this setup and calls
    `pyperclip.set_clipboard("wl-clipboard")` to ensure proper clipboard
    functionality.

    Args:
        mock_grabclipboard: Mock for `ImageGrab.grabclipboard`.
        mock_mocr: Mock for the `MangaOcr` class.
        mock_set_clipboard: Mock for `pyperclip.set_clipboard`.
        mock_os_system: Mock for `os.system` to simulate the presence of
            `wl-copy`.
        monkeypatch: The pytest fixture for modifying environment variables.
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
def test_run_handles_clipboard_non_image_error(mock_logger_warning, mock_sleep, mock_grabclipboard, mock_mocr):
    """Tests that `run` gracefully handles a non-image clipboard error.

    This test simulates the specific `OSError` with the message "cannot
    identify image file", which is a common occurrence on Linux when the
    clipboard content is not an image. It verifies that the `run` function
    correctly ignores this error when not in verbose mode, preventing
    unnecessary warnings from being logged.

    Args:
        mock_logger_warning: Mock for the logger's warning method.
        mock_sleep: Mock for `time.sleep` to prevent delays.
        mock_grabclipboard: Mock for `ImageGrab.grabclipboard` to simulate
            the specific OSError.
        mock_mocr: Mock for the `MangaOcr` class.
    """
    mock_grabclipboard.side_effect = [OSError("cannot identify image file"), KeyboardInterrupt]

    with pytest.raises(KeyboardInterrupt):
        run(read_from="clipboard", write_to="clipboard", verbose=False)

    mock_logger_warning.assert_not_called()


@patch("manga_ocr.run.MangaOcr")
@patch("pathlib.Path.iterdir")
@patch("time.sleep", side_effect=KeyboardInterrupt)
@patch("loguru.logger.warning")
def test_run_in_directory_mode_ignores_subdirectories(
    mock_logger_warning, mock_sleep, mock_iterdir, mock_mocr, tmp_path
):
    """Tests that directory monitoring mode correctly ignores subdirectories.

    This test verifies that the `run` function, when in directory monitoring
    mode, correctly processes only files and ignores any subdirectories it
    encounters. This prevents errors and ensures that only valid image files
    are passed to the OCR engine.

    Args:
        mock_logger_warning: Mock for the logger's warning method.
        mock_sleep: Mock for `time.sleep` to prevent delays.
        mock_iterdir: Mock for `pathlib.Path.iterdir` to simulate finding
            both a file and a directory.
        mock_mocr: Mock for the `MangaOcr` class.
        tmp_path: The pytest fixture for a temporary directory.
    """
    mock_mocr_instance = mock_mocr.return_value
    mock_mocr_instance.return_value = "test ocr"

    # Create a dummy image file and a subdirectory
    img_path = tmp_path / "img.png"
    Image.new("RGB", (100, 100)).save(img_path)
    subdir_path = tmp_path / "subdir"
    subdir_path.mkdir()

    output_file = tmp_path / "output.txt"

    # Simulate directory scanning
    mock_iterdir.side_effect = [
        [],  # Initial scan is empty
        [img_path, subdir_path],  # Second scan finds both items
    ]

    with pytest.raises(KeyboardInterrupt):
        run(read_from=str(tmp_path), write_to=str(output_file))

    # Verify that OCR was called only for the image file
    mock_mocr_instance.assert_called_once()

    # Verify that no warning was logged for the subdirectory
    mock_logger_warning.assert_not_called()


def test_run_raises_on_invalid_read_from_path(tmp_path):
    """Tests that `run` raises an error for an invalid `read_from` path.

    This test ensures that the `run` function validates the `read_from` path
    and raises a `ValueError` if it is a file instead of a directory. This
    prevents the application from attempting to monitor a single file, which
    is not a supported operation.

    Args:
        tmp_path: The pytest fixture for a temporary directory.
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
def test_run_logs_warning_on_unopenable_image_file(mock_logger_warning, mock_sleep, mock_iterdir, mock_mocr, tmp_path):
    """Tests that `run` logs a warning for unopenable image files.

    This test verifies that when the `run` function encounters a file that
    cannot be opened by PIL, it logs a warning message. This is important for
    notifying the user of corrupted or invalid files in the monitored
    directory without crashing the application.

    Args:
        mock_logger_warning: Mock for the logger's warning method.
        mock_sleep: Mock for `time.sleep` to prevent delays.
        mock_iterdir: Mock for `pathlib.Path.iterdir` to simulate finding an
            invalid file.
        mock_mocr: Mock for the `MangaOcr` class.
        tmp_path: The pytest fixture for a temporary directory.
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
def test_run_handles_clipboard_no_png_error(mock_logger_warning, mock_sleep, mock_grabclipboard, mock_mocr):
    """Tests that `run` gracefully handles a missing PNG clipboard target.

    This test simulates the `OSError` with the message "target image/png not
    available", which can occur on Linux (X11) when the clipboard contains
    text instead of an image. It verifies that the `run` function correctly
    ignores this specific error when not in verbose mode, avoiding spurious
    warnings.

    Args:
        mock_logger_warning: Mock for the logger's warning method.
        mock_sleep: Mock for `time.sleep` to prevent delays.
        mock_grabclipboard: Mock for `ImageGrab.grabclipboard` to simulate
            the specific OSError.
        mock_mocr: Mock for the `MangaOcr` class.
    """
    mock_grabclipboard.side_effect = [OSError("target image/png not available"), KeyboardInterrupt]

    with pytest.raises(KeyboardInterrupt):
        run(read_from="clipboard", write_to="clipboard", verbose=False)

    mock_logger_warning.assert_not_called()