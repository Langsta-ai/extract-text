import os
from pathlib import Path
from pdf2image import convert_from_path
from openai import OpenAI
import base64
from PIL import Image
import io
from dotenv import load_dotenv
import time
from tenacity import retry, wait_exponential, stop_after_attempt
import numpy as np
import re

def optimize_image(image, max_size=(800, 800)):
    """Optimize image for OpenAI API"""
    image = image.convert('RGB')  # Ensure RGB format
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def image_to_base64(image, quality=75):
    """Convert image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode()

@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def process_image_with_openai(client, image):
    """Process image with OpenAI Vision API"""
    optimized_image = optimize_image(image)
    image_b64 = image_to_base64(optimized_image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Analyze the document and convert to markdown. For each image:
                    - Identify the person and context (e.g., 'Liz Jones', 'Bob Ayers')
                    - Provide correct coordinates of each particular image in the main image
                    - Don't split images of a particular content
                    - Use format: ![description](image_n.jpg){x:start_x,y:start_y,w:width,h:height,person:name}
                    - Generate a unique description for each image
                    - Ensure coordinates match the actual image content
                    - Coodinates should be relative to the main image
                    - Origin point is the top-left corner of the main image
                    - Content related images should be pointed to the content witout any overlap
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert this document to markdown, identifying each person's image and location precisely."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def extract_images_from_markdown(markdown_text, page_image, output_dir, page_num):
    """Extract images from markdown and save them"""
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Get original image dimensions
    orig_width, orig_height = page_image.size
    
    # Find image references with coordinates and person info
    pattern = r'!\[(.*?)\]\((.*?)\){x:(\d+),y:(\d+),w:(\d+),h:(\d+),person:(.*?)}'
    matches = re.finditer(pattern, markdown_text)
    
    updated_markdown = markdown_text
    for i, match in enumerate(matches, 1):
        desc, _, x, y, w, h, person = match.groups()
        x, y, w, h = map(int, (x, y, w, h))
        
        print(f"Extracting image for {person} at coordinates: x={x}, y={y}, w={w}, h={h}")
        
        try:
            # Extract and save image region
            image_region = page_image.crop((x, y, x+w, y+h))
            image_filename = f"page{page_num}_{person.lower().replace(' ', '_')}_image.jpg"
            image_path = images_dir / image_filename
            image_region.save(image_path, "JPEG", quality=95)
            
            # Update markdown with new image reference
            new_image_ref = f"![{desc}](images/{image_filename})"
            updated_markdown = updated_markdown.replace(match.group(0), new_image_ref)
            
            print(f"Extracted image for {person} at coordinates: x={x}, y={y}, w={w}, h={h}")
        except Exception as e:
            print(f"Error extracting image: {e}")
            continue
    
    return updated_markdown

def main():
    load_dotenv()
    pdf_path = Path("pages.pdf")
    output_dir = Path("output")
    
    # Check if PDF exists
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return
        
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not found in environment variables")
            return
            
        print(f"Converting PDF: {pdf_path}")
        pages = convert_from_path(pdf_path, dpi=72)
        total_pages = len(pages)
        
        if total_pages == 0:
            print("Error: No pages found in PDF")
            return
            
        print(f"Found {total_pages} pages to process...")
        
        for i, page in enumerate(pages, 1):
            start_time = time.time()
            print(f"\nProcessing page {i}/{total_pages}...")
            
            try:
                result = process_image_with_openai(client, page)
                if result:
                    # Extract and save images, get updated markdown
                    updated_markdown = extract_images_from_markdown(result, page, output_dir, i)
                    
                    # Save updated markdown
                    output_file = output_dir / f"page_{i}.md"
                    output_file.write_text(updated_markdown)
                    
                    processing_time = time.time() - start_time
                    print(f"✓ Completed page {i}/{total_pages} in {processing_time:.2f}s")
                else:
                    print(f"⚠️ No content generated for page {i}")
            except Exception as e:
                print(f"❌ Error processing page {i}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()