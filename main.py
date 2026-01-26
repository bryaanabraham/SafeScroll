from analyse_post import analyse_posts
from urllib.parse import urlparse
from datetime import datetime

def get_url():
    return

def extract_instagram_username(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return path.split("/")[0]

if __name__=="__main__":
    start = datetime.now()
    user_url= get_url()
    username = extract_instagram_username("https://www.instagram.com/maxverstappen1/")
    print(f"Getting data for user: {username}")
    result = analyse_posts(username, top_k=1)
    print("-"*50)
    print("-"*50)
    print(result)
    end = datetime.now()
    print(f"Execution time: {(end-start).total_seconds()}")