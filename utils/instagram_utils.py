"""
Instagram data fetching utilities
Optimized for parallel downloads and efficient session management
"""

import os
import instaloader
import requests
from pathlib import Path
from itertools import islice
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import config


def download_image(url: str, file_path: Path, session: requests.Session) -> Optional[str]:
    """
    Download a single image
    
    Args:
        url: Image URL
        file_path: Destination path
        session: Requests session for connection pooling
        
    Returns:
        File path if successful, None otherwise
    """
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        
        with open(file_path, "wb") as f:
            f.write(resp.content)
        
        return file_path.as_posix()
    except Exception:
        return None


def fetch_instagram_posts(
    target_username: str,
    login_username: Optional[str] = None,
    login_password: Optional[str] = None,
    top_k: int = 1,
    download_dir: str = config.DOWNLOAD_DIR
) -> List[Dict]:
    """
    Fetch Instagram posts with parallel image downloading
    
    Args:
        target_username: Instagram username to fetch posts from
        login_username: Optional login username
        login_password: Optional login password
        top_k: Number of posts to fetch
        download_dir: Directory to save images
        
    Returns:
        List of post dictionaries with caption, metadata, and image paths
    """
    # Create base download directory
    base_path = Path(download_dir) / target_username
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize Instaloader
    loader = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        quiet=True  # Suppress output
    )
    
    # Login if credentials provided
    if login_username and login_password:
        loader.login(login_username, login_password)
    
    # Get profile
    profile = instaloader.Profile.from_username(loader.context, target_username)
    
    posts_data = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    # Process posts
    for idx, post in enumerate(islice(profile.get_posts(), top_k), start=1):
        caption = post.caption
        metadata = post._node
        
        # Collect image URLs
        image_urls = []
        if post.typename == "GraphImage":
            image_urls.append(post.url)
        elif post.typename == "GraphSidecar":
            for node in post.get_sidecar_nodes():
                image_urls.append(node.display_url)
        
        # Create post folder
        post_folder = base_path / f"post_{idx}_{post.shortcode}"
        post_folder.mkdir(parents=True, exist_ok=True)
        
        # Download images in parallel
        image_paths = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for img_idx, url in enumerate(image_urls, start=1):
                file_path = post_folder / f"image_{img_idx}.jpg"
                future = executor.submit(download_image, url, file_path, session)
                futures[future] = (url, file_path)
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    image_paths.append(result)
        
        posts_data.append({
            "caption": caption,
            "metadata": metadata,
            "images": image_paths
        })
    
    return posts_data
