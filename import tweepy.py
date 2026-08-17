import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import polars as pl
import requests
from bs4 import BeautifulSoup

# Authenticate with the Twitter API
def authenticate_twitter_app(consumer_key, consumer_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    return tweepy.API(auth)

# Fetch and enhance Twitter data
def fetch_enhanced_twitter_data(api, query, items_limit=10):
    tweets = []
    for tweet in tweepy.Cursor(api.search, q=query, tweet_mode='extended').items(items_limit):
        tweets.append({
            'id': tweet.id_str,
            'text': tweet.full_text,
            'created_at': str(tweet.created_at),
            'likes': tweet.favorite_count,
            'retweets': tweet.retweet_count,
            'has_links': 'http' in tweet.full_text,
            'link': extract_link(tweet.full_text) if 'http' in tweet.full_text else None,
            'has_images': 'media' in tweet.entities,
            'sentiment': TextBlob(tweet.full_text).sentiment.polarity
        })

    return pl.DataFrame(tweets)

# Extract the first link from a tweet text
def extract_link(text):
    words = text.split()
    for word in words:
        if word.startswith('http'):
            return word
    return None

# Analyze the link content using BeautifulSoup
def analyze_link_content(link):
    if link:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.title.string if soup.title else 'No title found'
    return 'No link'

# Generate and save a word cloud
def generate_and_save_word_cloud(df, filename='word_cloud.png'):
    all_text = ' '.join(df.get_column('text').to_list())
    word_cloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

# Main analysis function
def main():
    # Twitter API credentials
    CONSUMER_KEY = 'your_consumer_key'
    CONSUMER_SECRET = 'your_consumer_secret'
    ACCESS_TOKEN = 'your_access_token'
    ACCESS_TOKEN_SECRET = 'your_access_secret'

    api = authenticate_twitter_app(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    twitter_data = fetch_enhanced_twitter_data(api, "Python", items_limit=100)
    print(twitter_data.head())

    # Analyzing link content and updating the DataFrame
    twitter_data['link_content'] = twitter_data['link'].apply(analyze_link_content)

    # Generate and save word cloud
    generate_and_save_word_cloud(twitter_data)

    return twitter_data

if __name__ == '__main__':
    df = main()
