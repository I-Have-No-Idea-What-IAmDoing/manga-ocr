import fire

from manga_ocr.run import run


def main():
    """The main entry point for the command-line interface.

    This function serves as the entry point when the `manga_ocr` package is
    executed as a script. It uses the `fire` library to expose the `run`
    function from `manga_ocr.run` to the command line, allowing users to
    easily run the OCR process and configure its behavior with various
    arguments.
    """
    fire.Fire(run)


if __name__ == "__main__":
    main()