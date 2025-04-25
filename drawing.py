from PIL import ImageFont
import utils
import textwrap

# Function to add text to speech bubbles with proper wrapping
def add_text_to_bubble(draw, box, text, font_path="arial.ttf"):
    x, y, w, h = box
    
    # Add padding inside the bubble
    x, y, w, h = utils.apply_padding(x, y, w, h, 10)
    
    # Estimate font size based on bubble height
    # Start with a reasonable size and adjust if needed
    font_size = 16  # Initial guess
    min_font_size = 12  # Don't go smaller than this
    
    font = None
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
        font_size = 12
    
    # Calculate how many characters can fit per line based on width
    avg_char_width = font.getlength("A")  # Approximate width of a character
    chars_per_line = max(1, int(w / avg_char_width))
    
    # Wrap text
    wrapped_text = textwrap.fill(text, width=chars_per_line)
    lines = wrapped_text.split('\n')
    
    # If too many lines, reduce font size and try again
    while len(lines) * font_size > h and font_size > min_font_size:
        font_size -= 2
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        avg_char_width = font.getlength("A")
        chars_per_line = max(1, int(w / avg_char_width))
        wrapped_text = textwrap.fill(text, width=chars_per_line)
        lines = wrapped_text.split('\n')
    
    # Calculate vertical centering
    total_text_height = len(lines) * font_size
    start_y = y + (h - total_text_height) // 2
    
    # Draw each line of text
    for i, line in enumerate(lines):
        # Calculate horizontal centering for this line
        line_width = font.getlength(line)
        start_x = x + (w - line_width) // 2
        
        # Draw the text
        draw.text((start_x, start_y + i * font_size), line, fill=(0, 0, 0), font=font)
    
    return draw