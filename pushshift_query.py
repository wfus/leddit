import time

BASE_URL = "https://api.pushshift.io/reddit/search/comment/?"

url = BASE_URL
url += "q=AITA"
url += "&subreddit=amitheasshole"

print(url)

