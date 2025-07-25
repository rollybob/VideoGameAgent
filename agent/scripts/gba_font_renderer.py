
import os
from PIL import Image, ImageDraw

CHAR_WIDTH = 8
CHAR_HEIGHT = 8
CHARS_PER_ROW = 16

def render_text(text, font_image_path, output_path="rendered_text.png", spacing=1):
    # Load the font sheet
    font_sheet = Image.open(font_image_path).convert("RGBA")

    # Define basic ASCII mapping (0x20 to 0x7F)
    ascii_map = {chr(i): i - 32 for i in range(32, 127)}

    # Calculate canvas size
    num_chars = len(text)
    canvas_width = (CHAR_WIDTH + spacing) * num_chars
    canvas_height = CHAR_HEIGHT
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

    for i, char in enumerate(text):
        if char not in ascii_map:
            continue  # skip unknowns
        idx = ascii_map[char]
        row = idx // CHARS_PER_ROW
        col = idx % CHARS_PER_ROW
        glyph = font_sheet.crop((
            col * CHAR_WIDTH,
            row * CHAR_HEIGHT,
            (col + 1) * CHAR_WIDTH,
            (row + 1) * CHAR_HEIGHT
        ))
        canvas.paste(glyph, ((CHAR_WIDTH + spacing) * i, 0), glyph)

    canvas.save(output_path)
    print(f"[OK] Rendered image saved to: {output_path}")

# Example usage:
if __name__ == "__main__":
    render_text(
        "In the world which you are about to enter...",
        font_image_path="pokefirered.png"
    )
