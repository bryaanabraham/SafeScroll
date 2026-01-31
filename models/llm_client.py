"""
Optimized LLM Client for Gemini API
Handles all vision and text inference with improved prompts
Optimized for free tier usage
"""

from typing import List, Dict, Optional
import config
from google import genai
from google.genai import types
from PIL import Image
import os
import time
from dotenv import load_dotenv
load_dotenv()


class LLMClient:
    """Handles all LLM interactions with Gemini API"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=self.api_key)
        self.vision_model = "gemini-2.5-flash-lite-preview-09-2025"
        self.text_model = "gemini-2.5-flash-lite-preview-09-2025"
        
        self.request_delay = 3.0
        self.max_retries = 5  
        self.retry_delay = 5.0
    
    def _call_with_retry(self, func, *args, **kwargs):
        """
        Wrapper to handle API calls with retry logic for rate limits
        """
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    time.sleep(delay)
                
                result = func(*args, **kwargs)
                
                # Add delay after successful request to respect rate limits
                time.sleep(self.request_delay)
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        print(f"⚠️  Rate limit hit. Waiting {wait_time:.0f}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"Rate limit exceeded after {self.max_retries} retries. Please wait and try again.")
                
                # For other errors, raise immediately
                raise e
        
        raise Exception(f"Failed after {self.max_retries} attempts")
    
    def extract_time_of_day(self, image_paths: List[str]) -> str:
        """
        Determine approximate time of day from image(s)
        
        Improved prompt for better accuracy
        """
        try:
            # Load images as PIL Image objects
            imgs = [Image.open(img) for img in image_paths]
            
            # Create multimodal content
            contents = [
                "Analyze the lighting and shadows in the image(s) to determine the approximate time of day. "
                "Respond ONLY with a time range in format 'HH:MM AM/PM - HH:MM AM/PM' (e.g., '2:00 PM - 4:00 PM'). "
                "Consider: natural light color temperature, shadow length and direction, sky brightness. "
                "If uncertain, respond with 'Unable to determine'."
            ] + imgs

            def _api_call():
                return self.client.models.generate_content(
                    model=self.vision_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=50  # Reduced for free tier
                    )
                )
            
            response = self._call_with_retry(_api_call)
            return response.text.strip()
        except Exception as e:
            return f"Unable to determine time: {str(e)}"
    
    def summarize_single_image(self, image_path: str) -> str:
        """
        Generate detailed analysis for a single image
        
        Analyzes one image at a time for individual image summaries
        """
        try:
            img = Image.open(image_path)
            
            contents = [
                "Provide a detailed analysis of this image covering:\n"
                "1. Main subjects/people (appearance, actions, expressions)\n"
                "2. Setting/location (indoor/outdoor, environmental details)\n"
                "3. Objects and items present\n"
                "4. Activities or events occurring\n"
                "5. Mood, atmosphere, and overall context\n\n"
                "Be specific and observant. Use clear, concise language. Keep response under 150 words."
            ] + [img]
            
            def _api_call():
                return self.client.models.generate_content(
                    model=self.vision_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.5,
                        max_output_tokens=300  # Reduced for single image
                    )
                )
            
            response = self._call_with_retry(_api_call)
            return response.text.strip()
        except Exception as e:
            return f"Unable to analyze image: {str(e)}"
    
    def extract_text_from_image(self, image_paths: List[str]) -> str:
        """
        Extract all visible text using context-aware OCR
        
        Improved prompt for better text extraction
        """
        try:
            imgs = [Image.open(img) for img in image_paths]
            
            contents = [
                "Extract ALL visible text from the image(s).\n"
                "Include:\n"
                "- Signs, labels, banners\n"
                "- Printed text on objects\n"
                "- Digital displays or screens\n"
                "- Handwritten text if legible\n"
                "- Watermarks or logos with text\n\n"
                "Format the output clearly. If no text is visible, respond with 'No text detected'."
            ] + imgs
            
            def _api_call():
                return self.client.models.generate_content(
                    model=self.vision_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=800  # Reduced for free tier
                    )
                )
            
            response = self._call_with_retry(_api_call)
            return response.text.strip()
        except Exception as e:
            return f"No text detected: {str(e)}"
    
    def extract_geolocation(self, image_paths: List[str]) -> Dict[str, Optional[float]]:
        """
        Extract geolocation (latitude, longitude, altitude) from image(s)
        
        Note: This analyzes visual cues like landmarks, signs, etc.
        For EXIF GPS data, use a dedicated EXIF reader instead.
        """
        try:
            imgs = [Image.open(img) for img in image_paths]
            
            contents = [
                "Analyze the image(s) and attempt to determine the geographic location.\n"
                "Look for:\n"
                "- Recognizable landmarks, buildings, or monuments\n"
                "- Street signs, location names, or city identifiers\n"
                "- Geographic features (mountains, bodies of water, terrain)\n"
                "- Language on signs or cultural indicators\n"
                "- Any visible coordinates or location markers\n\n"
                "Respond in this EXACT format:\n"
                "Latitude: [number or 'Unknown']\n"
                "Longitude: [number or 'Unknown']\n"
                "Altitude: [number in meters or 'Unknown']\n"
                "Location: [best guess of location name]\n"
                "Confidence: [Low/Medium/High]\n\n"
                "If you cannot determine any coordinates, still provide best guess for location name."
            ] + imgs
            
            def _api_call():
                return self.client.models.generate_content(
                    model=self.vision_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=200  # Reduced for free tier
                    )
                )
            
            response = self._call_with_retry(_api_call)
            
            # Parse response
            result_text = response.text.strip()
            result = {
                "latitude": None,
                "longitude": None,
                "altitude": None,
                "location": "Unknown",
                "confidence": "Unknown",
                "raw_response": result_text
            }
            
            # Simple parsing of the response
            for line in result_text.split('\n'):
                line = line.strip()
                if line.lower().startswith('latitude:'):
                    try:
                        lat_str = line.split(':', 1)[1].strip()
                        if lat_str.lower() != 'unknown':
                            result["latitude"] = float(lat_str)
                    except:
                        pass
                elif line.lower().startswith('longitude:'):
                    try:
                        lon_str = line.split(':', 1)[1].strip()
                        if lon_str.lower() != 'unknown':
                            result["longitude"] = float(lon_str)
                    except:
                        pass
                elif line.lower().startswith('altitude:'):
                    try:
                        alt_str = line.split(':', 1)[1].strip().replace('meters', '').replace('m', '').strip()
                        if alt_str.lower() != 'unknown':
                            result["altitude"] = float(alt_str)
                    except:
                        pass
                elif line.lower().startswith('location:'):
                    result["location"] = line.split(':', 1)[1].strip()
                elif line.lower().startswith('confidence:'):
                    result["confidence"] = line.split(':', 1)[1].strip()
            
            return result
            
        except Exception as e:
            return {
                "latitude": None,
                "longitude": None,
                "altitude": None,
                "location": "Unknown",
                "confidence": "Unknown",
                "error": str(e)
            }
    
    def analyze_caption(self, caption: str) -> str:
        """
        Analyze the post caption
        """
        if not caption or caption.strip() == "":
            return "No caption provided."
        
        try:
            contents = f"Analyze this Instagram caption and provide insights about its tone, message, and purpose (max 100 words):\n\n{caption}"
            
            def _api_call():
                return self.client.models.generate_content(
                    model=self.text_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.6,
                        max_output_tokens=200
                    )
                )
            
            response = self._call_with_retry(_api_call)
            return response.text.strip()
        except Exception as e:
            return f"Unable to analyze caption: {str(e)}"
    
    def generate_structured_summary(
        self,
        image_analyses: List[Dict[str, str]],
        caption_analysis: str,
        face_data: str = ""
    ) -> str:
        """
        Generate structured summary in the format:
        Image 1: [analysis]
        Image 2: [analysis]
        Caption: [caption analysis]
        
        Args:
            image_analyses: List of dicts with 'image_path' and 'analysis'
            caption_analysis: Analysis of the caption
            face_data: Optional face detection data
        """
        try:
            # Build the structured output
            output_lines = []
            
            # Add image analyses
            for idx, img_data in enumerate(image_analyses, 1):
                analysis = img_data.get('analysis', 'No analysis available')
                output_lines.append(f"Image {idx}: {analysis}")
            
            # Add caption analysis
            output_lines.append(f"\nCaption: {caption_analysis}")
            
            # Add face data if available
            if face_data and face_data.strip():
                output_lines.append(f"\n--- Face Detection ---\n{face_data.strip()}")
            
            return "\n\n".join(output_lines)
            
        except Exception as e:
            return f"Error generating structured summary: {str(e)}"
    
    def generate_comprehensive_summary(
        self,
        time_content: str,
        summary_content: str,
        text_content: str,
        caption_content: Optional[str],
        face_data: str,
        geolocation_data: Optional[Dict] = None
    ) -> str:
        """
        Synthesize all information into a comprehensive analysis
        
        Improved prompt for better synthesis
        """
        try:
            # Build structured input
            geo_section = ""
            if geolocation_data:
                geo_section = f"""
## Geolocation Data
Location: {geolocation_data.get('location', 'Unknown')}
Coordinates: {geolocation_data.get('latitude', 'N/A')}, {geolocation_data.get('longitude', 'N/A')}
Altitude: {geolocation_data.get('altitude', 'N/A')} meters
Confidence: {geolocation_data.get('confidence', 'Unknown')}
"""
            
            analysis_input = f"""# Instagram Post Analysis

## Temporal Information
{time_content}

## Visual Content Summary
{summary_content}

## Extracted Text
{text_content}

## User Caption
{caption_content if caption_content else 'No caption provided'}

## Face Detection Data
{face_data if face_data else 'No faces detected'}
{geo_section}
---

Based on the above information, provide a comprehensive analysis addressing:

1. **What is happening**: Describe the main activity or event
2. **Location & Time Context**: Where and when this appears to be taking place
3. **People Involved**: Who is present and what can be inferred about them
4. **Key Messages**: Important text, messages, or themes conveyed
5. **Overall Assessment**: What story does this post tell? What is its purpose or significance?

Synthesize all available information into a coherent narrative. Be analytical but concise.
"""
            
            def _api_call():
                return self.client.models.generate_content(
                    model=self.text_model,
                    contents=analysis_input,
                    config=types.GenerateContentConfig(
                        temperature=0.6,
                        max_output_tokens=1200  # Reduced for free tier
                    )
                )
            
            response = self._call_with_retry(_api_call)
            return response.text.strip()
        except Exception as e:
            return f"Error generating comprehensive summary: {str(e)}"


# Global singleton instance
_llm_client_instance: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client instance"""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance
