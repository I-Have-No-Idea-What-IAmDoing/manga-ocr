"""The manga_ocr package provides an easy-to-use interface for OCR on manga.

This package encapsulates the functionality for loading the OCR model and
running inference on images. The main entry point is the `MangaOcr` class,
which is exposed at the top level of the package for convenient access.

Example:
    >>> from manga_ocr import MangaOcr
    >>> mocr = MangaOcr()
    >>> text = mocr('image.jpg')
    >>> print(text)
"""

from ._version import __version__ as __version__
from manga_ocr.ocr import MangaOcr as MangaOcr
