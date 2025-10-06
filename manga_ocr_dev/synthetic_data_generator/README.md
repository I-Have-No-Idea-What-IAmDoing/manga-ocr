# Synthetic Data Generator

This tool generates synthetic image-text pairs that mimic the appearance of Japanese manga. It is designed to create a diverse and realistic dataset for training the `manga-ocr` model.

## Features

-   **High-Quality Text Rendering**: Uses the `pictex` library for robust and accurate text rendering, supporting complex layouts.
-   **Flexible Text Sources**: Can generate text randomly from a given vocabulary or use text from a provided corpus.
-   **Advanced Text Layouts**:
    -   Vertical and horizontal text rendering.
    -   Multi-line text with smart line-breaking.
    -   **Furigana** (ruby text) added randomly to kanji.
    -   **Tate-Chu-Yoko** (horizontal-in-vertical) for ASCII text.
-   **Rich Styling Options**:
    -   Supports a wide variety of fonts with configurable sizing.
    -   Applies text effects like strokes and glows.
    -   Draws high-contrast speech bubbles around text.
-   **Realistic Backgrounds**: Overlays text onto a diverse set of background images, with augmentations like blur and contrast adjustments to increase variety.

## Generation Pipeline

The data generation process follows these steps:

1.  **Text Processing**: A line of text is either generated randomly or taken from a corpus. It is then processed to add random furigana to kanji and apply Tate-Chu-Yoko to ASCII characters.
2.  **Text Rendering**: The processed text, including all markup, is rendered into a transparent RGBA image using the `pictex` library. This approach provides fine-grained control over the text's appearance, including font, size, color, and effects.
3.  **Image Composition**:
    -   The rendered text image is optionally enclosed in a speech bubble.
    -   A random background is selected and augmented.
    -   The text image (with its bubble) is then composited onto the background.
4.  **Final Touches**: A final smart crop is performed to ensure the text is well-positioned and legible, producing the final training image.