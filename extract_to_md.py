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
    image_b64 = image_to_base64(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Analyze the document and convert to markdown. For each image:
                    - Find objects which are mentioned in the text (e.g., 'machines', 'plants', 'building', 'people').
                    - Find the representation of each object, ensuring the whole form of the object is centered within the rectangle, in the image.
                    - Provide the exact rectangle coordinates of each object in the image.
                    - Provide a image-name of each object with unique number in the format: image_{number}.jpg.
                    - Use the format for each object in the image: ![description](image-name.jpg){x:start_x,y:start_y,w:width,h:height,object:name}.
                    - Ensure coordinates match the actual image content.
                    - The origin point is the top-left corner of the main image.
                    - Provide a unique description for each object.
                    - If no objects are found, indicate that explicitly.
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert this document to markdown, identifying each object's image and location precisely."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=4096
        )
        
        # Log the entire response for debugging
        # print("API Response:", response)
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def extract_images_from_markdown(markdown_text: str, page_image: Image.Image, output_dir: Path, page_num: int):
    """Extract images from markdown and save them"""
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Get original image dimensions
    orig_width, orig_height = page_image.size
    print(f"Original image size: {orig_width}x{orig_height}")
    
    # Find image references with coordinates and object info
    pattern = r'!\[(.*?)\]\((image_\d+\.jpg)\){x:(\d+),y:(\d+),w:(\d+),h:(\d+)'
    matches = list(re.finditer(pattern, markdown_text))  # Convert to list
    
    print(f"Found {len(matches)} matches in markdown.")
    
    updated_markdown = markdown_text
    for i, match in enumerate(matches, 1):
        desc, image_filename, x, y, w, h = match.groups()
        x, y, w, h = map(int, (x, y, w, h))
        
        # Log original coordinates
        print(f"Original coordinates for {image_filename}: x={x}, y={y}, w={w}, h={h}")
        
        # Check if coordinates are within the bounds of the original image
        if x < 0 or y < 0 or x + w > orig_width or y + h > orig_height:
            print(f"⚠️ Invalid cropping coordinates for {image_filename}: x={x}, y={y}, w={w}, h={h} (original size: {orig_width}x{orig_height})")
            continue
        
        img_num = re.search(r'image_(\d+)\.jpg', image_filename).group(1)
        new_filename = f"page_{page_num}_image_{img_num}.jpg"    
        
        print(f"Extracting image for {new_filename} at coordinates: x={x}, y={y}, w={w}, h={h}")
        
        try:
            # Extract and save image region
            image_region = page_image.crop((x, y, x+w, y+h))
            image_path = images_dir / new_filename
            image_region.save(image_path, "JPEG", params={"quality": 95, "optimize": True})
            
            # Update markdown with new image reference using new_filename
            new_image_ref = f"![{desc}](images/{new_filename})"
            updated_markdown = updated_markdown.replace(match.group(0), new_image_ref)
            
            print(f"Extracted image for {new_filename} at coordinates: x={x}, y={y}, w={w}, h={h}")
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
        pages = convert_from_path(pdf_path)
        total_pages = len(pages)
        
        if total_pages == 0:
            print("Error: No pages found in PDF")
            return
            
        print(f"Found {total_pages} pages to process...")
        
        for i, page in enumerate(pages, 1):
            start_time = time.time()
            print(f"\nProcessing page {i}/{total_pages}...")
            
            try:
                optimized_image = optimize_image(page, max_size=(1280, 1280))
                
                # Process markdown only
                result = process_image_with_openai(client, optimized_image)
                if result:
                    updated_markdown = extract_images_from_markdown(result, optimized_image, output_dir, i)
                    output_file = output_dir / f"page_{i}.md"
                    output_file.write_text(updated_markdown)
                
                # Save original page image
                output_image = output_dir / f"page_{i}.jpg"
                page.save(output_image, "JPEG", quality=95)
                
                processing_time = time.time() - start_time
                print(f"✓ Completed page {i}/{total_pages} in {processing_time:.2f}s")
            except Exception as e:
                print(f"❌ Error processing page {i}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()