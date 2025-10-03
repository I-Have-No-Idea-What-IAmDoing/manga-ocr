"""A vendored copy of the html2image library.

This package is a third-party library included directly in the project to
ensure stability and avoid dependency conflicts. It is used by the synthetic
data generator to render HTML and CSS into images, which is essential for
creating styled text for the training dataset.

For the original library, see: https://github.com/vladkens/html2image
"""

from .html2image import Html2Image