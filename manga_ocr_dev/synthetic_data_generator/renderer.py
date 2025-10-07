"""Image rendering engine for the synthetic data generator.

This script defines the `Renderer` class, which is responsible for turning
styled text into images. It uses the `html2image` library to render HTML and
CSS, allowing for complex text layouts, including vertical text and furigana.
"""

import os
import uuid
import threading
import tempfile
from pathlib import Path
from textwrap import dedent
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import cv2
import numpy as np
from manga_ocr_dev.vendored.html2image import Html2Image


def crop_by_alpha(img, margin=0):
    """Crops an image by removing transparent padding."""
    y, x = np.where(img[:, :, 3] > 0)
    if y.size == 0 or x.size == 0:
        return np.zeros((1, 1, 4), dtype=img.dtype)
    ymin, ymax = np.min(y), np.max(y)
    xmin, xmax = np.min(x), np.max(x)
    img = img[ymin : ymax + 1, xmin : xmax + 1]
    if margin > 0:
        img = np.pad(img, ((margin, margin), (margin, margin), (0, 0)))
    return img


def get_css(
    font_size, font_path, vertical=True, background_color="white",
    text_color="black", glow_size=0, glow_color="black", stroke_size=0,
    stroke_color="black", letter_spacing=None, line_height=0.5,
    text_orientation=None,
):
    """Generates a CSS string for styling text rendered in a browser."""
    styles = [
        f"background-color: {background_color};",
        f"font-size: {font_size}px;",
        f"color: {text_color};",
        "font-family: custom;",
        f"line-height: {line_height};",
        "margin: 20px;",
    ]
    if text_orientation:
        styles.append(f"text-orientation: {text_orientation};")
    if vertical:
        styles.append("writing-mode: vertical-rl;")
    if glow_size > 0:
        styles.append(f"text-shadow: 0 0 {glow_size}px {glow_color};")
    if stroke_size > 0:
        shadows = [f"{x}px {y}px 0 {stroke_color}" for x in range(-stroke_size, stroke_size + 1) for y in range(-stroke_size, stroke_size + 1) if x != 0 or y != 0]
        styles.extend([f"text-shadow: {','.join(shadows)};", "-webkit-font-smoothing: antialiased;"])
    if letter_spacing:
        styles.append(f"letter-spacing: {letter_spacing}em;")
    path = Path(font_path)
    if not path.is_absolute():
        path = path.absolute()
    font_uri = path.as_uri()
    styles_str = "\n".join(styles)
    css = f'@font-face {{font-family: custom; src: url("{font_uri}");}}\n'
    css += f"html, body {{\n{styles_str}\n}}"
    return css


class Renderer:
    """Renders text into images using `html2image`."""

    def __init__(self, cdp_port=9222, browser_executable=None, debug=False):
        """Initializes the Renderer.

        Args:
            cdp_port (int, optional): The port for the Chrome DevTools Protocol.
            browser_executable (str | None, optional): The path to the browser executable.
            debug (bool, optional): If True, enables additional debugging features.
        """
        self.debug = debug
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

        flags = [
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
            "--no-zygote", "--ozone-platform=headless", "--disable-sync",
            "--disable-login-screen-apps", "--disable-default-apps",
            "--disable-infobars", "--disable-notifications", "--disable-extensions",
            "--disable-background-networking", "--disable-component-update",
            "--disable-client-side-phishing-detection", "--disable-domain-reliability",
            "--disable-popup-blocking", "--disable-hang-monitor",
            "--disable-features=TranslateUI",
            f"--user-data-dir={os.path.join(self.temp_dir.name, 'user-data')}",
            "--disable-gcm", "--remote-allow-origins=*",
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

    def __enter__(self):
        self.hti.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hti.__exit__(exc_type, exc_val, exc_tb)
        self.executor.shutdown(wait=False)
        self.temp_dir.cleanup()

    def render(self, lines, override_css_params=None):
        """Renders the given lines of text into a styled image.

        Args:
            lines (list[str]): A list of strings, where each string is a line of text.
            override_css_params (dict, optional): A dictionary of CSS parameters.

        Returns:
            A tuple containing:
                - np.ndarray: The rendered text image as a BGRA NumPy array.
                - dict: The dictionary of the CSS parameters used for rendering.
        """
        with self.lock:
            img, params = self._render_text(lines, override_css_params)
        return img, params

    def _render_text(self, lines, override_css_params=None):
        """Renders text with CSS styling on a transparent background."""
        params = self.get_random_css_params()
        if override_css_params:
            params.update(override_css_params)

        css = get_css(**params)

        if not lines or not "".join(lines):
            return None, params

        size = (
            int(max(len(line) for line in lines) * params["font_size"] * 1.5),
            int(len(lines) * params["font_size"] * (3 + params["line_height"])),
        )
        if params["vertical"]:
            size = size[::-1]

        lines_str = "\n".join([f"<p>{line}</p>" for line in lines])
        html = f'<html><head><meta charset="UTF-8"><style>{css}</style></head><body>{lines_str}</body></html>'
        html = dedent(html)

        if self.debug:
            params["html"] = html

        html_filename = str(uuid.uuid4()) + ".html"
        img_bytes = None
        try:
            self.hti.load_str(html, as_filename=html_filename)
            future = self.executor.submit(self.hti.screenshot_as_bytes, file=html_filename, size=size)
            try:
                img_bytes = future.result(timeout=30)
            except TimeoutError:
                print(f"Skipping render for '{''.join(lines)[:30]}...' due to timeout.")
                future.cancel()
                return None, params
            except Exception as e:
                print(f"Screenshot failed with an exception: {e}")
                return None, params
        finally:
            temp_file_path = os.path.join(self.hti.temp_path, html_filename)
            if os.path.exists(temp_file_path):
                try:
                    self.hti._remove_temp_file(html_filename)
                except Exception as e:
                    print(f"Error removing temp file: {e}")

        if img_bytes is None:
            return None, params

        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, params
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        return crop_by_alpha(img), params

    @staticmethod
    def get_random_css_params():
        """Generates a dictionary of random CSS parameters for text rendering."""
        params = {
            "font_size": np.random.randint(48, 96),
            "vertical": np.random.rand() < 0.7,
            "line_height": np.random.uniform(1.4, 2.0),
            "background_color": "transparent",
            "text_color": "black" if np.random.rand() < 0.7 else "white",
        }
        if np.random.rand() < 0.7:
            params["text_orientation"] = "upright"
        if np.random.rand() < 0.2:
            params["letter_spacing"] = np.random.uniform(-0.05, 0.1)
        effect = np.random.choice(["stroke", "glow", "none"], p=[0.4, 0.15, 0.45])
        if effect == "stroke":
            params["stroke_size"] = np.random.choice([1, 2, 3])
            params["stroke_color"] = "white" if params["text_color"] == "black" else "black"
        elif effect == "glow":
            params["glow_size"] = np.random.choice([2, 5, 10])
            params["glow_color"] = "white" if np.random.rand() < 0.8 else "black"
        return params