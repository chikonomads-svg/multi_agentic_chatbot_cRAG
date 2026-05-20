"""
Image Text Overlay Module
Adds readable, well-formatted text to AI-generated slide backgrounds using PIL/Pillow.
"""

import base64
import io
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional
import os


class SlideTextOverlay:
    """Adds professional text overlays to slide images."""
    
    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height
        
    def create_slide_with_text(
        self, 
        background_image_b64: str,
        slide_title: str,
        content_items: List[str],
        slide_number: int,
        total_slides: int = 10
    ) -> str:
        """
        Create a professional slide with text overlay.
        
        Args:
            background_image_b64: Base64 encoded background image
            slide_title: Title of the slide
            content_items: List of content bullet points
            slide_number: Current slide number
            total_slides: Total number of slides
            
        Returns:
            Base64 encoded image with text overlay
        """
        # Decode background image
        img_data = base64.b64decode(background_image_b64)
        img = Image.open(io.BytesIO(img_data))
        
        # Resize if needed
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        
        # Create drawing context
        draw = ImageDraw.Draw(img)
        
        # Try to load a nice font, fall back to default if not available
        try:
            # Try system fonts
            title_font = ImageFont.truetype("arial.ttf", 56)
            subtitle_font = ImageFont.truetype("arial.ttf", 32)
            content_font = ImageFont.truetype("arial.ttf", 28)
            small_font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                # Try Linux/Mac fonts
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
                subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
                content_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                # Fall back to default
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
                content_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
        
        # Add semi-transparent overlay at top for title
        overlay_height = 180
        # Create a small overlay strip and then composite it onto a full-size transparent layer
        top_strip = Image.new('RGBA', (self.width, overlay_height), (0, 0, 0, 120))
        # Create a full-size transparent overlay to match the base image size (required by alpha_composite)
        full_overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        # Paste the semi-transparent top strip onto the full overlay at the top
        full_overlay.paste(top_strip, (0, 0), top_strip)
        img = Image.alpha_composite(img.convert('RGBA'), full_overlay)
        draw = ImageDraw.Draw(img)
        
        # Add slide counter at top right
        counter_text = f"{slide_number}/{total_slides}"
        bbox = draw.textbbox((0, 0), counter_text, font=small_font)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (self.width - text_width - 30, 30),
            counter_text,
            font=small_font,
            fill=(255, 255, 255, 200)
        )
        
        # Add slide title at top
        title_y = 60
        
        # Draw title with shadow for readability
        shadow_offset = 2
        draw.text(
            (self.width//2 + shadow_offset, title_y + shadow_offset),
            slide_title,
            font=title_font,
            fill=(0, 0, 0, 180),
            anchor="mm"
        )
        draw.text(
            (self.width//2, title_y),
            slide_title,
            font=title_font,
            fill=(255, 255, 255, 255),
            anchor="mm"
        )
        
        # Add content items in the middle/lower portion
        content_start_y = 280
        line_spacing = 60
        left_margin = 80
        right_margin = 80
        max_width = self.width - left_margin - right_margin
        
        for i, item in enumerate(content_items[:5]):  # Max 5 items
            y_position = content_start_y + (i * line_spacing)
            
            # Draw bullet point
            bullet = "•"
            draw.text(
                (left_margin, y_position),
                bullet,
                font=content_font,
                fill=(255, 255, 255, 255)
            )
            
            # Wrap text if too long
            wrapped_text = self._wrap_text(item, content_font, max_width - 40, draw)
            
            # Draw text with shadow
            for line_idx, line in enumerate(wrapped_text):
                line_y = y_position + (line_idx * 36)
                
                # Shadow
                draw.text(
                    (left_margin + 35 + shadow_offset, line_y + shadow_offset),
                    line,
                    font=content_font,
                    fill=(0, 0, 0, 150)
                )
                # Main text
                draw.text(
                    (left_margin + 35, line_y),
                    line,
                    font=content_font,
                    fill=(255, 255, 255, 255)
                )
        
        # Add footer with branding
        footer_y = self.height - 50
        draw.text(
            (self.width//2, footer_y),
            "2026 Trending Topics",
            font=small_font,
            fill=(255, 255, 255, 150),
            anchor="mm"
        )
        
        # Convert to RGB for saving
        final_img = img.convert('RGB')
        
        # Save to bytes
        buffer = io.BytesIO()
        final_img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Return base64
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.Draw) -> List[str]:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]


def add_text_to_slide(
    background_b64: str,
    slide_title: str,
    content: Any,
    slide_number: int,
    total_slides: int = 10
) -> str:
    """
    Convenience function to add text to a slide.
    
    Args:
        background_b64: Base64 encoded background image
        slide_title: Title of the slide
        content: Content items (list or string)
        slide_number: Current slide number
        total_slides: Total number of slides
        
    Returns:
        Base64 encoded image with text overlay
    """
    overlay = SlideTextOverlay()
    
    # Convert content to list if string
    if isinstance(content, str):
        content_items = [content]
    elif isinstance(content, list):
        content_items = [str(item) for item in content]
    else:
        content_items = [str(content)]
    
    return overlay.create_slide_with_text(
        background_b64,
        slide_title,
        content_items,
        slide_number,
        total_slides
    )
