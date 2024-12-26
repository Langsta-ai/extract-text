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

def optimize_image(image, max_size=(800, 800)):
    """Optimize image for OpenAI API"""
    image = image.convert('RGB')
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def image_to_base64(image, quality=75):
    """Convert image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode()

@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def process_html_with_openai(client, image):
    """Process image with OpenAI Vision API for HTML conversion"""
    image_b64 = image_to_base64(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Convert the document to semantic HTML with these requirements:
                    1. Use semantic HTML5 tags (header, main, section, article, etc.)
                    2. Preserve text content and formatting
                    3. Structure the document with proper layout blocks
                    4. Use CSS classes for styling
                    5. For images use: <img src="image_{number}.jpg" alt="description">
                    
                    Return ONLY valid HTML without any additional text or explanation.
                    The HTML should be ready to insert into a template.
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert this document to HTML, preserving layout and content."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=4096
        )
        
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("Empty response from API")
            
        # Basic validation of the response
        if not ('<' in content and '>' in content):
            raise ValueError(f"Invalid HTML response: {content[:100]}...")
            
        return content.strip()
        
    except Exception as e:
        print(f"Error in process_html_with_openai: {str(e)}")
        print(f"Full error details: {type(e).__name__}: {str(e)}")
        raise

def save_html_output(html_content: str, images_dir: Path, page_num: int, output_dir: Path):
    """Save HTML content with proper styling and image references"""
    html_template = """{content}"""
    
    try:
        # Update image references in HTML
        html_content = html_content.replace('src="image_', f'src="images/page_{page_num}_image_')
        
        # Save HTML file
        html_file = output_dir / f"page_{page_num}.html"
        formatted_html = html_template.format(page_num=page_num, content=html_content)
        html_file.write_text(formatted_html, encoding='utf-8')
        
    except Exception as e:
        print(f"Error saving HTML: {str(e)}")
        raise

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
                
                # Process HTML
                html_result = process_html_with_openai(client, optimized_image)
                if html_result:
                    save_html_output(html_result, output_dir / "images", i, output_dir)
                
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