"""Image rendering engine for the synthetic data generator.

This script defines the `Renderer` class, which is responsible for turning
styled text into images. It uses the `html2image` library to render HTML and
CSS, allowing for complex text layouts, including vertical text and furigana.
The renderer also handles compositing the text onto various backgrounds,
applying text bubbles, and adding other visual effects to create a diverse
and realistic dataset for training the OCR model.
"""

import os
import base64
import uuid
import threading
import tempfile
from pathlib import Path
from textwrap import dedent
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import albumentations as A
import cv2
import numpy as np
from manga_ocr_dev.vendored.html2image import Html2Image

from manga_ocr_dev.env import BACKGROUND_DIR
from manga_ocr_dev.synthetic_data_generator.utils import get_background_df


class Renderer:
    """Renders text into images with various styles, backgrounds, and effects.

    This class is a core component of the synthetic data generation pipeline.
    It uses `html2image` to render HTML/CSS styled text into images, which
    are then used to train the OCR model. The renderer can add random
    backgrounds, text bubbles, and other visual effects to create a diverse
    and realistic dataset.

    Attributes:
        hti (Html2Image): An instance of `Html2Image` for rendering HTML.
        lock (threading.Lock): A lock to ensure thread-safe rendering, as
            `html2image` may not be thread-safe.
        background_df (pd.DataFrame): A DataFrame containing paths and metadata
            for available background images.
        max_size (int): The maximum size (in pixels) of the longest side of the
            output image.
    """

    def __init__(self, cdp_port=9222, browser_executable=None, debug=False):
        """Initializes the Renderer.

        Args:
            cdp_port (int, optional): The port for the Chrome DevTools Protocol,
                used by `html2image` to control the browser. Defaults to 9222.
            browser_executable (str | None, optional): The path to the browser
                executable (e.g., Chrome, Chromium). If None, `html2image`
                will attempt to find a default installation. Defaults to None.
            debug (bool, optional): If True, enables additional debugging
                features, such as browser logging. Defaults to False.
        """
        self.debug = debug
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

        flags = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--no-zygote",
            "--ozone-platform=headless",
            "--disable-sync",
            "--disable-login-screen-apps",
            "--disable-default-apps",
            "--disable-infobars",
            "--disable-notifications",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-client-side-phishing-detection",
            "--disable-domain-reliability",
            "--disable-popup-blocking",
            "--disable-hang-monitor",
            "--disable-features=TranslateUI",
            f"--user-data-dir={os.path.join(self.temp_dir.name, 'user-data')}",
            "--disable-gcm",
            "--remote-allow-origins=*",
        ]
        if not self.debug:
            flags.append("--disable-logging")

        self.hti = Html2Image(
            browser="chrome-cdp",
            browser_cdp_port=cdp_port,
            browser_executable=browser_executable,
            temp_path=self.temp_dir.name,
            custom_flags=flags,
        )
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.background_df = get_background_df(BACKGROUND_DIR)
        self.max_size = 600

    def __enter__(self):
        """Starts the Html2Image instance as a context manager."""
        self.hti.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exits the Html2Image context manager, cleaning up resources."""
        self.hti.__exit__(exc_type, exc_val, exc_tb)
        self.executor.shutdown(wait=False)
        self.temp_dir.cleanup()

    def render(self, lines, override_css_params=None):
        """Renders the given lines of text into a styled, synthetic image.

        This is the main rendering method. It orchestrates the process of
        rendering text with CSS, adding a background, and applying final
        transformations to create a complete synthetic image.

        Args:
            lines (list[str]): A list of strings, where each string is a line
                of text to be rendered.
            override_css_params (dict, optional): A dictionary of CSS
                parameters to override the default rendering styles. This allows
                for fine-grained control over the output. Defaults to None.

        Returns:
            A tuple containing:
                - np.ndarray: The final rendered image as a grayscale NumPy array.
                - dict: A dictionary of the CSS parameters used for rendering.
        """
        params = self.get_random_css_params()
        if override_css_params:
            params.update(override_css_params)

        # Select and prepare a background image
        background_path = self.background_df.sample(1).iloc[0].path
        background = cv2.imread(background_path)

        # Apply augmentations to the background
        t = [
            A.HorizontalFlip(),
            A.RandomRotate90(),
            A.RandomBrightnessContrast((-0.2, 0.4), (-0.8, -0.3), p=0.8),
            A.Blur((3, 5), p=0.3),
        ]
        background = A.Compose(t)(image=background)["image"]

        # Encode the augmented background to a base64 data URI
        _, buffer = cv2.imencode(".png", background)
        bg_base64 = base64.b64encode(buffer).decode("utf-8")
        params["background_image_data_uri"] = f"data:image/png;base64,{bg_base64}"

        # Render HTML with text on top of the background image
        with self.lock:
            img, params = self._render_html(lines, params)

        if img is None:
            return None, params

        # Final random crop to make the framing less predictable
        h, w, _ = img.shape
        if h > 10 and w > 10:
            target_h = int(h * np.random.uniform(0.8, 1.0))
            target_w = int(w * np.random.uniform(0.8, 1.0))
            img = A.RandomCrop(height=target_h, width=target_w)(image=img)["image"]

        img = A.LongestMaxSize(self.max_size)(image=img)["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, params

    def _render_html(self, lines, params):
        """Renders text with CSS styling directly onto a background image.

        This method generates HTML and CSS to render the text, including any
        text bubbles, directly on a background image in a single step using
        `html2image`.

        Args:
            lines (list[str]): A list of strings to be rendered.
            params (dict): A dictionary of CSS parameters, including the path
                to the background image.

        Returns:
            A tuple containing:
                - np.ndarray: The rendered BGR image.
                - dict: The dictionary of CSS parameters used for rendering.
        """
        css = get_css(**params)

        if not lines or not "".join(lines):
            return None, params

        # Estimate a suitable size for the rendering surface.
        # This is a rough guess; the final image is cropped anyway.
        size = (
            int(max(len(line) for line in lines) * params["font_size"] * 1.5),
            int(len(lines) * params["font_size"] * (3 + params["line_height"])),
        )
        if params["vertical"]:
            size = size[::-1]
        size = (max(size[0], 500), max(size[1], 500))

        lines_str = "\n".join([f"<p>{line}</p>" for line in lines])
        html = f"""\
        <html>
        <head>
          <meta charset="UTF-8">
          <style>{css}</style>
        </head>
        <body>{lines_str}</body>
        </html>
        """
        html = dedent(html)

        if self.debug:
            params["html"] = html

        html_filename = str(uuid.uuid4()) + ".html"
        img_bytes = None
        try:
            self.hti.load_str(html, as_filename=html_filename)

            # Run screenshot in a separate thread with a timeout
            future = self.executor.submit(
                self.hti.screenshot_as_bytes, file=html_filename, size=size
            )
            try:
                img_bytes = future.result(timeout=30)  # 30-second timeout
            except TimeoutError:
                print(f"Skipping render for '{''.join(lines)[:30]}...' due to timeout.")
                future.cancel()
                return None, params
            except Exception as e:
                print(f"Screenshot failed with an exception: {e}")
                return None, params

        finally:
            # Ensure the temporary HTML file is always removed
            temp_file_path = os.path.join(self.hti.temp_path, html_filename)
            if os.path.exists(temp_file_path):
                try:
                    self.hti._remove_temp_file(html_filename)
                except Exception as e:
                    print(f"Error removing temp file: {e}")

        if img_bytes is None:
            return None, params

        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, params

        return img, params

    @staticmethod
    def get_random_css_params():
        """Generates a dictionary of random CSS parameters for text rendering.

        This method creates a set of randomized CSS properties to introduce
        variety into the synthetic data. It randomizes properties like font
        size, orientation (vertical/horizontal), and text effects like
        strokes or glows.

        Returns:
            A dictionary of randomly generated CSS parameters.
        """
        params = {
            "font_size": np.random.randint(48, 96),
            "vertical": np.random.rand() < 0.7,
            "line_height": np.random.uniform(1.4, 2.0),
            "text_color": "black" if np.random.rand() < 0.7 else "white",
            "draw_bubble": np.random.rand() < 0.7,
        }

        if np.random.rand() < 0.7:
            params["text_orientation"] = "upright"
        if np.random.rand() < 0.2:
            params["letter_spacing"] = np.random.uniform(-0.05, 0.1)

        effect = np.random.choice(
            ["stroke", "glow", "none"], p=[0.4, 0.15, 0.45]
        )
        if effect == "stroke":
            params["stroke_size"] = np.random.choice([1, 2, 3])
            params["stroke_color"] = "white" if params["text_color"] == "black" else "black"
        elif effect == "glow":
            params["glow_size"] = np.random.choice([2, 5, 10])
            params["glow_color"] = "white" if np.random.rand() < 0.8 else "black"

        if params["draw_bubble"]:
            params["bubble_padding"] = np.random.randint(15, 30)
            params["bubble_border_radius"] = np.random.randint(10, 40)
            params["bubble_border_width"] = np.random.randint(2, 5)
            if params["text_color"] == "black":
                params["bubble_background_color"] = "white"
                params["bubble_border_color"] = "black"
            else:
                params["bubble_background_color"] = "black"
                params["bubble_border_color"] = "white"
        else:
            params["bubble_background_color"] = "transparent"

        return params


def get_css(
    font_size,
    font_path,
    vertical=True,
    text_color="black",
    glow_size=0,
    glow_color="black",
    stroke_size=0,
    stroke_color="black",
    letter_spacing=None,
    line_height=0.5,
    text_orientation=None,
    background_image_data_uri=None,
    draw_bubble=False,
    bubble_background_color="white",
    bubble_padding=20,
    bubble_border_radius=20,
    bubble_border_width=3,
    bubble_border_color="black",
):
    """Generates a CSS string for styling text rendered in a browser.

    This function constructs a CSS string that can be used to style text in
    an HTML document. It includes properties for font size, font family,
    writing mode (vertical/horizontal), colors, and various text effects like
    glows and strokes.

    Args:
        font_size (int): The font size in pixels.
        font_path (str): The path to the font file to embed using `@font-face`.
        vertical (bool, optional): If True, sets writing mode to vertical-rl.
        background_color (str, optional): The background color of the text.
        text_color (str, optional): The color of the text.
        glow_size (int, optional): The size of the text glow in pixels.
        glow_color (str, optional): The color of the text glow.
        stroke_size (int, optional): The size of the text stroke in pixels,
            simulated using multiple text shadows.
        stroke_color (str, optional): The color of the text stroke.
        letter_spacing (float, optional): The letter spacing in 'em' units.
        line_height (float, optional): The line height as a multiple of font size.
        text_orientation (str, optional): The text orientation (e.g., 'upright').

    Returns:
        The generated CSS string.
    """
    body_styles = [
        f"font-size: {font_size}px;",
        f"color: {text_color};",
        "font-family: custom;",
        f"line-height: {line_height};",
        "margin: 20px;",
        "display: inline-block;",
    ]

    if text_orientation:
        body_styles.append(f"text-orientation: {text_orientation};")

    if vertical:
        body_styles.append("writing-mode: vertical-rl;")

    if glow_size > 0:
        body_styles.append(f"text-shadow: 0 0 {glow_size}px {glow_color};")

    if stroke_size > 0:
        shadows = []
        for x in range(-stroke_size, stroke_size + 1):
            for y in range(-stroke_size, stroke_size + 1):
                if x != 0 or y != 0:
                    shadows.append(f"{x}px {y}px 0 {stroke_color}")
        body_styles.extend(
            [
                "text-shadow: " + ",".join(shadows) + ";",
                "-webkit-font-smoothing: antialiased;",
            ]
        )

    if letter_spacing:
        body_styles.append(f"letter-spacing: {letter_spacing}em;")

    if draw_bubble:
        body_styles.append(f"background-color: {bubble_background_color};")
        body_styles.append(f"padding: {bubble_padding}px;")
        body_styles.append(f"border-radius: {bubble_border_radius}px;")
        body_styles.append(f"border: {bubble_border_width}px solid {bubble_border_color};")
        body_styles.append("box-shadow: 5px 5px 15px rgba(0,0,0,0.2);")
    else:
        body_styles.append("background-color: transparent;")

    # Convert the font path to a file URI for the browser to load it correctly
    path = Path(font_path)
    if not path.is_absolute():
        path = path.absolute()
    font_uri = path.as_uri()

    html_styles = [
        "margin: 0; padding: 0;",
        "display: flex;",
        "justify-content: center;",
        "align-items: center;",
        "width: 100vw; height: 100vh;",
    ]
    if background_image_data_uri:
        html_styles.append(f'background-image: url("{background_image_data_uri}");')
        html_styles.append("background-size: cover;")
        html_styles.append("background-position: center;")
    else:
        html_styles.append("background-color: white;")

    body_styles_str = "\n".join(body_styles)
    html_styles_str = "\n".join(html_styles)
    css = f'@font-face {{font-family: custom; src: url("{font_uri}");}}\n'
    css += f"html {{\n{html_styles_str}\n}}\n"
    css += f"body {{\n{body_styles_str}\n}}"
    return css