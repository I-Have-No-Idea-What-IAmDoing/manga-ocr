import fire

from manga_ocr.run import run


def main():
    """Entry point for the command-line interface.

    This function uses `fire` to expose the `run` function to the command line,
    allowing users to run the OCR process with various arguments.
    """
    fire.Fire(run)


if __name__ == "__main__":
    main()