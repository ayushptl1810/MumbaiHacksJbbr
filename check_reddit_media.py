import praw
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

post_url = 'https://reddit.com/r/NoFilterNews/comments/1p9a2kx/absolutely_unacceptable_trump_sparks_outrage_with/'
submission = reddit.submission(url=post_url)

print(f"Post title: {submission.title}")
print(f"Post URL: {submission.url}")
print(f"Has image: {'i.redd.it' in submission.url or 'imgur' in submission.url}")
print(f"URL type: {type(submission.url)}")
