"""Image rendering engine for the synthetic data generator.

This script defines the `Renderer` class, which is responsible for turning
styled text into images. It uses the `html2image` library to render HTML and
CSS, allowing for complex text layouts, including vertical text and furigana.
The renderer also handles compositing the text onto various backgrounds,
applying text bubbles, and adding other visual effects to create a diverse
and realistic dataset for training the OCR model.
"""

import os
import uuid
import threading
import tempfile
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

    def __init__(self, cdp_port=9222, browser_executable=None):
        """Initializes the Renderer.

        Args:
            cdp_port (int, optional): The port for the Chrome DevTools Protocol,
                used by `html2image` to control the browser. Defaults to 9222.
            browser_executable (str | None, optional): The path to the browser
                executable (e.g., Chrome, Chromium). If None, `html2image`
                will attempt to find a default installation. Defaults to None.
        """
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.hti = Html2Image(
            browser="chrome-cdp",
            browser_cdp_port=cdp_port,
            browser_executable=browser_executable,
            temp_path=self.temp_dir.name,
            custom_flags=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-zygote",
                "--ozone-platform=headless",
                "--disable-sync",
                "--disable-login-screen-apps",
                "--disable-logging",
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
            ],
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
        with self.lock:
            img, params = self.render_text(lines, override_css_params)
        if img is None:
            return np.zeros((100, 100), dtype=np.uint8), params

        img = self.render_background(img, params)

        # render_background can return an empty image on failure, which will crash albumentations
        if img.size == 0:
            return np.zeros((100, 100), dtype=np.uint8), params

        img = A.LongestMaxSize(self.max_size)(image=img)["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, params

    def render_text(self, lines, override_css_params=None):
        """Renders text with CSS styling on a transparent background.

        This method takes lines of text, generates random CSS styles (with
        optional overrides), and uses `html2image` to render the text as a
        BGRA image with a transparent background. This image can then be
        composited onto a background.

        Args:
            lines (list[str]): A list of strings to be rendered.
            override_css_params (dict, optional): A dictionary of CSS
                parameters to override the default styles. Defaults to None.

        Returns:
            A tuple containing:
                - np.ndarray: The rendered text as a BGRA image.
                - dict: The dictionary of CSS parameters used for rendering.
        """

        params = self.get_random_css_params()
        if override_css_params:
            params.update(override_css_params)

        css = get_css(**params)

        # This is just a rough estimate; the image is cropped later anyway.
        if not lines or not "".join(lines):
            return None, params

        size = (
            int(max(len(line) for line in lines) * params["font_size"] * 1.5),
            int(len(lines) * params["font_size"] * (3 + params["line_height"])),
        )
        if params["vertical"]:
            size = size[::-1]

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

        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:  # imdecode can fail
            return None, params

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

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
            "font_size": np.random.randint(36, 60),
            "vertical": np.random.rand() < 0.7,
            "line_height": np.random.uniform(0.4, 0.8),
            "background_color": "transparent",
            "text_color": "black" if np.random.rand() < 0.7 else "white",
        }

        if np.random.rand() < 0.7:
            params["text_orientation"] = "upright"
        if np.random.rand() < 0.2:
            params["letter_spacing"] = np.random.uniform(-0.05, 0.1)

        effect = np.random.choice(
            ["stroke", "glow", "none"], p=[0.8, 0.15, 0.05]
        )
        if effect == "stroke":
            params["stroke_size"] = np.random.choice([1, 2, 3, 4])
            params["stroke_color"] = "white" if params["text_color"] == "black" else "black"
        elif effect == "glow":
            params["glow_size"] = np.random.choice([2, 5, 10])
            params["glow_color"] = "white" if np.random.rand() < 0.8 else "black"

        return params

    def render_background(self, img, params):
        """Adds a background and optionally a text bubble to an image.

        This method takes a BGRA image with text on a transparent background,
        composites it onto a randomly selected background image, and can also
        draw a text bubble around the text. The final image is cropped and
        returned as a BGR image.

        Args:
            img (np.ndarray): The input BGRA image with a transparent background.
            params (dict): A dictionary of rendering parameters, used to
                determine bubble color and other style aspects.

        Returns:
            The final BGR image with text composited onto a background.
        """
        draw_bubble = np.random.random() < 0.7

        img = crop_by_alpha(img, margin=0)
        if min(img.shape[:2]) < 10:
            return np.zeros((10, 10, 3), dtype=np.uint8)

        m0 = int(min(img.shape[:2]) * np.random.uniform(0.2, 0.4))
        img = np.pad(img, ((m0, m0), (m0, m0), (0, 0)))

        background_path = self.background_df.sample(1).iloc[0].path
        background = cv2.imread(background_path)

        t = [
            A.HorizontalFlip(),
            A.RandomRotate90(),
            A.InvertImg(),
            A.RandomBrightnessContrast(
                (-0.2, 0.4), (-0.8, -0.3), p=0.5 if draw_bubble else 1
            ),
            A.Blur((3, 5), p=0.3),
            A.Resize(img.shape[0], img.shape[1]),
        ]
        background = A.Compose(t)(image=background)["image"]

        if not draw_bubble:
            if params["text_color"] == "white":
                img[:, :, :3] = 255 - img[:, :, :3]
        else:
            bubble = self.create_bubble(img.shape, m0, params)
            background = blend(bubble, background)

        img = blend(img, background)

        ymin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        ymax = img.shape[0] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmax = img.shape[1] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        img = img[ymin:ymax, xmin:xmax]
        return img

    def create_bubble(self, shape, margin, params):
        """Creates a distorted, rounded rectangle to serve as a text bubble.

        Args:
            shape (tuple): The shape of the target image for the bubble.
            margin (int): The base margin for positioning the bubble.
            params (dict): A dictionary of rendering parameters, used to
                determine bubble color.

        Returns:
            An RGBA NumPy array containing the generated text bubble.
        """
        radius = np.random.uniform(0.7, 1.0)
        thickness = np.random.choice([1, 2, 3])
        alpha = np.random.randint(60, 100)
        sigma = np.random.randint(10, 15)

        ymin = margin - int(min(shape[:2]) * np.random.uniform(0.07, 0.12))
        ymax = shape[0] - margin + int(min(shape[:2]) * np.random.uniform(0.07, 0.12))
        xmin = margin - int(min(shape[:2]) * np.random.uniform(0.07, 0.12))
        xmax = shape[1] - margin + int(min(shape[:2]) * np.random.uniform(0.07, 0.12))

        bubble = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)

        if params["text_color"] == "black":
            bubble_fill_color = (255, 255, 255, 255)
            bubble_border_color = (0, 0, 0, 255)
        else:
            bubble_fill_color = (0, 0, 0, 255)
            bubble_border_color = (255, 255, 255, 255)

        bubble = rounded_rectangle(
            bubble, (xmin, ymin), (xmax, ymax), radius=radius, color=bubble_fill_color, thickness=-1
        )
        bubble = rounded_rectangle(
            bubble, (xmin, ymin), (xmax, ymax), radius=radius, color=bubble_border_color, thickness=thickness
        )

        t = [A.ElasticTransform(alpha=alpha, sigma=sigma, p=0.8)]
        bubble = A.Compose(t)(image=bubble)["image"]
        return bubble


def crop_by_alpha(img, margin=0):
    """Crops an image by removing transparent padding.

    This function identifies the bounding box of all non-transparent pixels
    (where the alpha channel is > 0) and crops the image to this box. An
    optional margin can be added. If the image is fully transparent, a
    1x1 black image is returned to prevent errors.

    Args:
        img (np.ndarray): The input BGRA image as a NumPy array.
        margin (int, optional): The number of pixels to add as a margin
            around the cropped image. Defaults to 0.

    Returns:
        The cropped image as a BGRA NumPy array.
    """
    y, x = np.where(img[:, :, 3] > 0)
    if y.size == 0 or x.size == 0:
        return np.zeros((1, 1, 4), dtype=img.dtype)

    ymin, ymax = np.min(y), np.max(y)
    xmin, xmax = np.min(x), np.max(x)

    img = img[ymin : ymax + 1, xmin : xmax + 1]
    if margin > 0:
        img = np.pad(img, ((margin, margin), (margin, margin), (0, 0)))
    return img


def blend(img, background):
    """Blends a foreground image onto a background using alpha compositing.

    This function takes a foreground image with an alpha channel (BGRA) and
    blends it onto a background image (BGR). The alpha channel of the
    foreground determines the opacity of the blend.

    Args:
        img (np.ndarray): The foreground BGRA image.
        background (np.ndarray): The background BGR image.

    Returns:
        The blended BGR image as a NumPy array.
    """
    alpha = (img[:, :, 3] / 255)[:, :, np.newaxis]
    img = img[:, :, :3]
    img = (background * (1 - alpha) + img * alpha).astype(np.uint8)
    return img


def rounded_rectangle(
    src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA
):
    """Draws a rectangle with rounded corners on an image.

    This utility function, based on a solution from Stack Overflow, draws a
    customizable rounded rectangle. It's used to create text bubbles.

    Args:
        src (np.ndarray): The source image to draw on.
        top_left (tuple[int, int]): The (x, y) of the top-left corner.
        bottom_right (tuple[int, int]): The (x, y) of the bottom-right corner.
        radius (float, optional): The corner radius as a fraction of the
            smaller side. Defaults to 1.
        color (tuple or int, optional): The color of the rectangle.
        thickness (int, optional): The thickness of the outline. A negative
            value fills the rectangle. Defaults to 1.
        line_type (int, optional): The line type for drawing (e.g., `cv2.LINE_AA`).

    Returns:
        The source image with the rounded rectangle drawn on it.
    """
    p1, p3 = top_left, bottom_right
    p2, p4 = (p3[0], p1[1]), (p1[0], p3[1])
    height, width = abs(p3[1] - p1[1]), abs(p3[0] - p1[0])
    corner_radius = int(min(height, width) / 2 * radius)

    if thickness < 0:
        # Draw filled rectangles
        cv2.rectangle(src, (p1[0] + corner_radius, p1[1]), (p3[0] - corner_radius, p3[1]), color, -1)
        cv2.rectangle(src, (p1[0], p1[1] + corner_radius), (p4[0] + corner_radius, p4[1] - corner_radius), color, -1)
        cv2.rectangle(src, (p2[0] - corner_radius, p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, -1)

    # Draw lines and arcs
    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
    cv2.line(src, (p4[0] + corner_radius, p4[1]), (p3[0] - corner_radius, p3[1]), color, abs(thickness), line_type)
    cv2.line(src, (p1[0], p1[1] + corner_radius), (p4[0], p4[1] - corner_radius), color, abs(thickness), line_type)

    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180, 0, 90, color, thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270, 0, 90, color, thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0, 0, 90, color, thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90, 0, 90, color, thickness, line_type)

    return src


def get_css(
    font_size,
    font_path,
    vertical=True,
    background_color="white",
    text_color="black",
    glow_size=0,
    glow_color="black",
    stroke_size=0,
    stroke_color="black",
    letter_spacing=None,
    line_height=0.5,
    text_orientation=None,
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
        shadows = []
        for x in range(-stroke_size, stroke_size + 1):
            for y in range(-stroke_size, stroke_size + 1):
                if x != 0 or y != 0:
                    shadows.append(f"{x}px {y}px 0 {stroke_color}")
        styles.extend([
            "text-shadow: " + ",".join(shadows) + ";",
            "-webkit-font-smoothing: antialiased;",
        ])

    if letter_spacing:
        styles.append(f"letter-spacing: {letter_spacing}em;")

    font_path = font_path.replace("\\", "/")
    styles_str = "\n".join(styles)
    css = f'@font-face {{font-family: custom; src: url("{font_path}");}}\n'
    css += f"body {{\n{styles_str}\n}}"
    return css