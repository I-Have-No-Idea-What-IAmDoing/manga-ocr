import fire

from manga_ocr.run import run


def main():
    """
    Main entry point of the script.
    """
    fire.Fire(run)


if __name__ == "__main__":
    main()