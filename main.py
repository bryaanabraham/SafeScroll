from datetime import datetime
from urllib.parse import urlparse

import config
from pipeline import analyze_posts


def extract_instagram_username(url: str) -> str:
    """Extract username from Instagram URL"""
    path = urlparse(url).path.strip("/")
    return path.split("/")[0]


def format_results(result: dict) -> str:
    """Format analysis results for display"""
    if not result.get("success"):
        return f"❌ Error: {result.get('error', 'Unknown error')}"
    
    output = []
    output.append("=" * 70)
    output.append(f"📊 Analysis Results for @{result['username']}")
    output.append(f"Posts analyzed: {result['posts_analyzed']}")
    output.append("=" * 70)
    output.append("")
    
    # Display each post summary
    for summary_data in result.get("summaries", []):
        output.append(f"📝 POST {summary_data['post_index']}")
        output.append("-" * 70)
        metadata = summary_data['metadata']
        output.append(f"Images: {metadata['num_images']} | Faces: {metadata['num_faces']}")
        output.append("")
        output.append(summary_data['summary'])
        output.append("")
        output.append("=" * 70)
        output.append("")
    
    return "\n".join(output)


if __name__ == "__main__":
    # Display device info
    config.DEVICE.print_info()
    print()
    
    # Start timing
    start_time = datetime.now()
    
    # Extract username
    username = extract_instagram_username("https://www.instagram.com/maxverstappen1/")
    print(f"🔍 Analyzing posts from: @{username}\n")
    
    # Run analysis
    result = analyze_posts(username, top_k=1)
    
    # Display results
    print(format_results(result))
    
    # Display timing
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"⏱️  Execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print()
