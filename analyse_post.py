from image_insights import get_time_from_image, summarize_image, context_aware_ocr, complete_summary
from detect_faces import detect_faces_from_image_path, draw_faces

def get_posts(target_username, login_username=None, login_password=None, top_k=1, download_dir="downloads"):
    """
    Extract top_k Instagram posts.
    Saves images locally and returns file paths.

    Returns: list of dicts with caption, metadata, images (file paths)
    """

    import os
    import instaloader
    import requests
    from itertools import islice
    from pathlib import Path

    # Create base download directory
    base_path = Path(download_dir) / target_username
    base_path.mkdir(parents=True, exist_ok=True)

    L = instaloader.Instaloader()

    if login_username and login_password:
        L.login(login_username, login_password)

    profile = instaloader.Profile.from_username(L.context, target_username)

    posts_data = []

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for idx, post in enumerate(islice(profile.get_posts(), top_k), start=1):

        caption = post.caption
        metadata = post._node  # internal structure (works but unofficial)

        image_urls = []

        if post.typename == "GraphImage":
            image_urls.append(post.url)

        elif post.typename == "GraphSidecar":
            for node in post.get_sidecar_nodes():
                image_urls.append(node.display_url)

        # Folder per post
        post_folder = base_path / f"post_{idx}_{post.shortcode}"
        post_folder.mkdir(parents=True, exist_ok=True)

        image_paths = []

        for img_idx, url in enumerate(image_urls, start=1):
            try:
                resp = session.get(url, timeout=10)
                resp.raise_for_status()

                file_path = post_folder / f"image_{img_idx}.jpg"

                with open(file_path, "wb") as f:
                    f.write(resp.content)

                image_paths.append(file_path.as_posix())

            except Exception as e:
                print(f"Failed to download {url}: {e}")

        posts_data.append({
            "caption": caption,
            "metadata": metadata,
            "images": image_paths
        })

    return posts_data

def analyse_posts(username, top_k=1):
    try:
        posts_data = get_posts(username, top_k=top_k)
        for post in posts_data:
            if post["images"] == []:
                continue
            else:
                image_time = get_time_from_image(post["images"])
                image_summary = summarize_image(post["images"])
                image_text = context_aware_ocr(post["images"])
                image_caption = post["caption"]
                faces = ''
                for image in post["images"]:
                    data = detect_faces_from_image_path(image)
                    faces += f"Faces' data in post:{post} -> {data}\n"
                summary = complete_summary(
                    time_content=image_time, 
                    summary_content=image_summary, 
                    text_content=image_text, 
                    caption_content=image_caption, 
                    face_data=faces
                )    
        return summary
    except Exception as e:
        print(f"Error occured: {e}")
        return

if __name__ == "__main__":
    username = "maxverstappen1"
    print(analyse_posts(username, top_k=1))