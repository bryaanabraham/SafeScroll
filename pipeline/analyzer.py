"""
Main analysis pipeline
Orchestrates Instagram post fetching, face detection, and LLM analysis
"""

from typing import Dict, Optional
from pathlib import Path

from models import get_face_detector, get_llm_client
from utils import fetch_instagram_posts, validate_image_paths


def analyze_posts(
    username: str,
    top_k: int = 1,
    login_username: Optional[str] = None,
    login_password: Optional[str] = None
) -> Dict:
    """
    Analyze Instagram posts for a given user
    
    Args:
        username: Instagram username
        top_k: Number of recent posts to analyze
        login_username: Optional login credentials
        login_password: Optional login credentials
        
    Returns:
        Dictionary containing analysis results and metadata
    """
    try:
        # Fetch posts
        posts_data = fetch_instagram_posts(
            target_username=username,
            login_username=login_username,
            login_password=login_password,
            top_k=top_k
        )
        
        if not posts_data:
            return {
                "success": False,
                "error": "No posts retrieved",
                "summary": None
            }
        
        # Get singleton instances (loaded once)
        face_detector = get_face_detector()
        llm_client = get_llm_client()
        
        # Process each post
        all_summaries = []
        
        for post_idx, post in enumerate(posts_data, 1):
            if not post["images"]:
                continue
            
            # Validate image paths
            valid_images = validate_image_paths(post["images"])
            if not valid_images:
                continue
            
            # Create optimized versions for different tasks
            # Use lower resolution for time/summary (faster processing)
            import config
            from utils.image_utils import create_llm_optimized_images, resize_image_for_llm
            
            llm_optimized = create_llm_optimized_images(
                valid_images,
                max_dimension=config.IMAGE_MAX_DIMENSION_LLM
            )
            
            # Use higher resolution for OCR (better accuracy)
            ocr_optimized = create_llm_optimized_images(
                valid_images,
                max_dimension=config.IMAGE_MAX_DIMENSION_OCR
            )
            
            # === Face Detection (Batch) ===
            face_results = face_detector.process_batch(valid_images)
            
            # Format face data
            face_data_str = ""
            for img_path, faces in face_results.items():
                img_name = Path(img_path).name
                if faces:
                    face_info = []
                    for i, face in enumerate(faces, 1):
                        info = f"Face {i}: confidence={face['det_score']:.2f}"
                        if face.get('age'):
                            info += f", age≈{face['age']}"
                        if face.get('gender') is not None:
                            gender = "Male" if face['gender'] == 1 else "Female"
                            info += f", gender={gender}"
                        face_info.append(info)
                    face_data_str += f"{img_name}: {len(faces)} face(s) - {'; '.join(face_info)}\n"
                else:
                    face_data_str += f"{img_name}: No faces detected\n"
            
            # === LLM Analysis ===
            # Analyze each image individually
            image_analyses = []
            for img_idx, img_path in enumerate(valid_images, 1):
                # Use downscaled version for analysis
                llm_img = llm_optimized[img_idx - 1] if img_idx <= len(llm_optimized) else img_path
                
                # Analyze this single image
                analysis = llm_client.summarize_single_image(llm_img)
                
                image_analyses.append({
                    'image_path': img_path,
                    'image_number': img_idx,
                    'analysis': analysis
                })
            
            # Analyze caption separately
            caption_text = post.get("caption", "")
            caption_analysis = llm_client.analyze_caption(caption_text)
            
            # Generate structured summary
            structured_summary = llm_client.generate_structured_summary(
                image_analyses=image_analyses,
                caption_analysis=caption_analysis,
                face_data=face_data_str
            )
            
            all_summaries.append({
                "post_index": post_idx,
                "summary": structured_summary,
                "metadata": {
                    "num_images": len(valid_images),
                    "num_faces": sum(len(faces) for faces in face_results.values()),
                    "has_caption": bool(post.get("caption")),
                    "image_analyses": image_analyses
                }
            })
        
        # Return structured results
        if all_summaries:
            return {
                "success": True,
                "username": username,
                "posts_analyzed": len(all_summaries),
                "summaries": all_summaries,
                "primary_summary": all_summaries[0]["summary"]  # For backward compatibility
            }
        else:
            return {
                "success": False,
                "error": "No valid images found in posts",
                "summary": None
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "summary": None
        }
