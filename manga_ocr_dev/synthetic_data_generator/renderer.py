import os
import uuid
import threading
import time

import albumentations as A
import cv2
import numpy as np
from manga_ocr_dev.vendored.html2image import Html2Image

from manga_ocr_dev.env import BACKGROUND_DIR
from manga_ocr_dev.synthetic_data_generator.utils import get_background_df


class Renderer:
    """Renders text into images with various styles, backgrounds, and effects.

    This class uses `html2image` to render HTML and CSS styled text into
    images. It can add random backgrounds, text bubbles, and other visual
    effects to create synthetic data for OCR. This is a core component of the
    synthetic data generation pipeline.

    Attributes:
        hti (Html2Image): An instance of `Html2Image` for rendering HTML.
        lock (threading.Lock): A lock to ensure thread-safe rendering, as
            `html2image` may not be thread-safe.
        background_df (pd.DataFrame): A DataFrame containing paths to available
            background images.
        max_size (int): The maximum size (in pixels) of the longest side of the
            output image.
    """
    def __init__(self, cdp_port=9222, browser_executable=None):
        """Initializes the Renderer.

        Args:
            cdp_port (int, optional): The port for the Chrome DevTools Protocol.
                Defaults to 9222.
            browser_executable (str | None, optional): The path to the browser
                executable. If None, `html2image` will try to find a default
                installation. Defaults to None.
        """
        self.hti = Html2Image(
            browser='chrome-cdp',
            browser_cdp_port=cdp_port,
            browser_executable=browser_executable,
            temp_path="/tmp/html2image",
            custom_flags=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu', '--no-zygote', '--ozone-platform=headless']
        )
        self.lock = threading.Lock()

        self.background_df = get_background_df(BACKGROUND_DIR)
        self.max_size = 600

    def __enter__(self):
        """Starts the context manager for the Html2Image instance."""
        self.hti.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exits the context manager for the Html2Image instance."""
        self.hti.__exit__(exc_type, exc_val, exc_tb)

    def render(self, lines, override_css_params=None):
        """Renders the given lines of text into a styled image.

        This is the main rendering method. It coordinates the process of
        rendering text with CSS, adding a background, and applying final
        transformations.

        Args:
            lines (list): A list of strings, where each string is a line of
                text to be rendered.
            override_css_params (dict, optional): A dictionary of CSS
                parameters to override the default rendering styles.
                Defaults to None.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The final rendered image as a grayscale NumPy
                  array.
                - dict: A dictionary of the CSS parameters used for rendering.
        """
        with self.lock:
            img, params = self.render_text(lines, override_css_params)
        img = self.render_background(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = A.LongestMaxSize(self.max_size)(image=img)["image"]
        return img, params

    def render_text(self, lines, override_css_params=None):
        """Renders text on a transparent background.

        This method takes lines of text, generates random CSS styles (with
        optional overrides), and uses `html2image` to render the text as a
        BGRA image with a transparent background.

        Args:
            lines (list): A list of strings to be rendered.
            override_css_params (dict, optional): A dictionary of CSS
                parameters to override the default styles. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The rendered text as a BGRA image.
                - dict: The CSS parameters used for rendering.
        """

        params = self.get_random_css_params()
        if override_css_params:
            params.update(override_css_params)

        css = get_css(**params)

        # this is just a rough estimate, image is cropped later anyway
        size = (
            int(max(len(line) for line in lines) * params["font_size"] * 1.5),
            int(len(lines) * params["font_size"] * (3 + params["line_height"])),
        )
        if params["vertical"]:
            size = size[::-1]
        html = self.lines_to_html(lines)

        # create a temporary file for the html content
        html_filename = str(uuid.uuid4()) + ".html"
        self.hti.load_str(html, as_filename=html_filename)

        # screenshot the temporary file and get the bytes
        img_bytes = self.hti.screenshot_as_bytes(
            file=html_filename,
            size=size,
        )

        # remove the temporary file
        self.hti._remove_temp_file(html_filename)

        # decode the bytes into an image
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

        # ensure image has 4 channels
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        return img, params

    @staticmethod
    def get_random_css_params():
        """Generates a dictionary of random CSS parameters for text rendering.

        This method creates a set of randomized CSS properties, such as font
        size, orientation, and text effects (stroke or shadow), to be used
        for rendering text.

        Returns:
            dict: A dictionary of CSS parameters.
        """
        params = {
            "font_size": 48,
            "vertical": True if np.random.rand() < 0.7 else False,
            "line_height": 0.5,
            "background_color": "transparent",
            "text_color": "black",
        }

        if np.random.rand() < 0.7:
            params["text_orientation"] = "upright"

        stroke_variant = np.random.choice(["stroke", "shadow", "none"], p=[0.8, 0.15, 0.05])
        if stroke_variant == "stroke":
            params["stroke_size"] = np.random.choice([1, 2, 3, 4, 8])
            params["stroke_color"] = "white"
        elif stroke_variant == "shadow":
            params["shadow_size"] = np.random.choice([2, 5, 10])
            params["shadow_color"] = ("white" if np.random.rand() < 0.8 else "black",)
        elif stroke_variant == "none":
            pass

        return params

    def render_background(self, img):
        """Adds a background and/or text bubble to an image.

        This method takes a BGRA image with a transparent background, adds a
        randomly selected background image, and optionally draws a text bubble
        around the text. The image is then cropped and returned as a BGR image.

        Args:
            img (np.ndarray): The input BGRA image with text on a transparent
                background.

        Returns:
            np.ndarray: The final BGR image with the background and other effects.
        """
        draw_bubble = np.random.random() < 0.7

        m0 = int(min(img.shape[:2]) * 0.3)
        img = crop_by_alpha(img, m0)

        background_path = self.background_df.sample(1).iloc[0].path
        background = cv2.imread(background_path)

        t = [
            A.HorizontalFlip(),
            A.RandomRotate90(),
            A.InvertImg(),
            A.RandomBrightnessContrast((-0.2, 0.4), (-0.8, -0.3), p=0.5 if draw_bubble else 1),
            A.Blur((3, 5), p=0.3),
            A.Resize(img.shape[0], img.shape[1]),
        ]

        background = A.Compose(t)(image=background)["image"]

        if not draw_bubble:
            if np.random.rand() < 0.5:
                img[:, :, :3] = 255 - img[:, :, :3]

        else:
            radius = np.random.uniform(0.7, 1.0)
            thickness = np.random.choice([1, 2, 3])
            alpha = np.random.randint(60, 100)
            sigma = np.random.randint(10, 15)

            ymin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            ymax = img.shape[0] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            xmin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            xmax = img.shape[1] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))

            bubble_fill_color = (255, 255, 255, 255)
            bubble_contour_color = (0, 0, 0, 255)
            bubble = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            bubble = rounded_rectangle(
                bubble,
                (xmin, ymin),
                (xmax, ymax),
                radius=radius,
                color=bubble_fill_color,
                thickness=-1,
            )
            bubble = rounded_rectangle(
                bubble,
                (xmin, ymin),
                (xmax, ymax),
                radius=radius,
                color=bubble_contour_color,
                thickness=thickness,
            )

            t = [
                A.ElasticTransform(alpha=alpha, sigma=sigma, p=0.8),
            ]
            bubble = A.Compose(t)(image=bubble)["image"]

            background = blend(bubble, background)

        img = blend(img, background)

        ymin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        ymax = img.shape[0] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmax = img.shape[1] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        img = img[ymin:ymax, xmin:xmax]
        return img

    def lines_to_html(self, lines):
        """Converts a list of text lines into an HTML string.

        Each line is wrapped in a `<p>` tag.

        Args:
            lines (list): A list of strings.

        Returns:
            str: The generated HTML string.
        """
        lines_str = "\n".join(["<p>" + line + "</p>" for line in lines])
        html = f"<html><body>\n{lines_str}\n</body></html>"
        return html

def min_or_zero(array):
    """Returns the minimum value of an array, or 0 if the array is empty.

    Args:
        array (np.ndarray): The input array.

    Returns:
        int or float: The minimum value or 0.
    """
    return np.min(array) if array.size > 0 else 0

def max_or_zero(array, default=0):
    """Returns the maximum value of an array, or a default value if empty.

    Args:
        array (np.ndarray): The input array.
        default (int or float, optional): The default value to return if the
            array is empty. Defaults to 0.

    Returns:
        int or float: The maximum value or the default value.
    """
    return np.max(array) if array.size > 0 else default

def crop_by_alpha(img, margin):
    """Crops an image based on its alpha channel.

    This function finds the bounding box of the non-transparent pixels in an
    image and crops the image to that box, with an added margin.

    Args:
        img (np.ndarray): The input BGRA image.
        margin (int): The margin to add around the cropped area.

    Returns:
        np.ndarray: The cropped image.
    """
    y, x = np.where(img[:, :, 3] > 0)
    ymin = min_or_zero(y)
    ymax = max_or_zero(y, default=img.shape[0])
    xmin = min_or_zero(x)
    xmax = max_or_zero(x, default=img.shape[1])

    if ymin >= ymax or xmin >= xmax:
        return np.zeros((margin * 2, margin * 2, 4), dtype=img.dtype)

    img = img[ymin : ymax + 1, xmin : xmax + 1]
    img = np.pad(img, ((margin, margin), (margin, margin), (0, 0)))
    return img


def blend(img, background):
    """Blends a foreground image onto a background image using alpha blending.

    Args:
        img (np.ndarray): The foreground BGRA image.
        background (np.ndarray): The background BGR image.

    Returns:
        np.ndarray: The blended BGR image.
    """
    alpha = (img[:, :, 3] / 255)[:, :, np.newaxis]
    img = img[:, :, :3]
    img = (background * (1 - alpha) + img * alpha).astype(np.uint8)
    return img


def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):
    """Draws a rounded rectangle on an image.

    This function is based on a solution from Stack Overflow.

    Args:
        src (np.ndarray): The source image.
        top_left (tuple): The top-left corner of the rectangle.
        bottom_right (tuple): The bottom-right corner of the rectangle.
        radius (float, optional): The radius of the corners as a fraction of
            the smaller side of the rectangle. Defaults to 1.
        color (tuple or int, optional): The color of the rectangle.
            Defaults to 255.
        thickness (int, optional): The thickness of the rectangle's outline.
            A negative value fills the rectangle. Defaults to 1.
        line_type (int, optional): The line type for drawing.
            Defaults to cv2.LINE_AA.

    Returns:
        np.ndarray: The image with the rounded rectangle drawn on it.
    """

    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_right[0], top_left[1])
    p3 = bottom_right
    p4 = (top_left[0], bottom_right[1])

    height = abs(bottom_right[1] - top_left[1])
    width = abs(bottom_right[0] - top_left[0])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (min(height, width) / 2))

    if thickness < 0:
        # big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
            [top_left_main_rect, bottom_right_main_rect],
            [top_left_rect_left, bottom_right_rect_left],
            [top_left_rect_right, bottom_right_rect_right],
        ]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(
        src,
        (p1[0] + corner_radius, p1[1]),
        (p2[0] - corner_radius, p2[1]),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p2[0], p2[1] + corner_radius),
        (p3[0], p3[1] - corner_radius),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p3[0] - corner_radius, p4[1]),
        (p4[0] + corner_radius, p3[1]),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p4[0], p4[1] - corner_radius),
        (p1[0], p1[1] + corner_radius),
        color,
        abs(thickness),
        line_type,
    )

    # draw arcs
    cv2.ellipse(
        src,
        (p1[0] + corner_radius, p1[1] + corner_radius),
        (corner_radius, corner_radius),
        180.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p2[0] - corner_radius, p2[1] + corner_radius),
        (corner_radius, corner_radius),
        270.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p3[0] - corner_radius, p3[1] - corner_radius),
        (corner_radius, corner_radius),
        0.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p4[0] + corner_radius, p4[1] - corner_radius),
        (corner_radius, corner_radius),
        90.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )

    return src


def get_css(
    font_size,
    font_path,
    vertical=True,
    background_color="white",
    text_color="black",
    shadow_size=0,
    shadow_color="black",
    stroke_size=0,
    stroke_color="black",
    letter_spacing=None,
    line_height=0.5,
    text_orientation=None,
):
    """Generates a CSS string for styling the rendered text.

    This function constructs a CSS string based on the provided parameters,
    including font properties, text orientation, colors, and effects like
    shadows and strokes.

    Args:
        font_size (int): The font size in pixels.
        font_path (str): The path to the font file.
        vertical (bool, optional): Whether to use vertical writing mode.
            Defaults to True.
        background_color (str, optional): The background color.
            Defaults to "white".
        text_color (str, optional): The text color. Defaults to "black".
        shadow_size (int, optional): The size of the text shadow.
            Defaults to 0.
        shadow_color (str, optional): The color of the text shadow.
            Defaults to "black".
        stroke_size (int, optional): The size of the text stroke.
            Defaults to 0.
        stroke_color (str, optional): The color of the text stroke.
            Defaults to "black".
        letter_spacing (float, optional): The letter spacing in 'em' units.
            Defaults to None.
        line_height (float, optional): The line height. Defaults to 0.5.
        text_orientation (str, optional): The text orientation (e.g.,
            'upright'). Defaults to None.

    Returns:
        str: The generated CSS string.
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

    if shadow_size > 0:
        styles.append(f"text-shadow: 0 0 {shadow_size}px {shadow_color};")

    if stroke_size > 0:
        # stroke is simulated by shadow overlaid multiple times
        styles.extend(
            [
                "text-shadow: " + ",".join([f"0 0 {stroke_size}px {stroke_color}"] * 10 * stroke_size) + ";",
                "-webkit-font-smoothing: antialiased;",
            ]
        )

    if letter_spacing:
        styles.append(f"letter-spacing: {letter_spacing}em;")

    font_path = font_path.replace("\\", "/")

    styles_str = "\n".join(styles)
    css = ""
    css += '\n@font-face {\nfont-family: custom;\nsrc: url("' + font_path + '");\n}\n'
    css += "body {\n" + styles_str + "\n}"
    return css