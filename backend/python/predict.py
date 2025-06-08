import sys
import json
import requests
from textblob import TextBlob

# Replace this with your actual News API key
NEWS_API_KEY = 'e2bd9a29e1cc4668964c6a26ce1dfe42'

def fetch_news(ticker):
    url = f'https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    data = response.json()

    if data.get('status') != 'ok':
        return [], f"Failed to fetch news: {data.get('message', 'Unknown error')}"

    articles = data.get('articles', [])
    headlines = [article['title'] for article in articles if article.get('title')]
    return headlines, None

def analyze_sentiment(headlines):
    if not headlines:
        return "Neutral", 0.0

    polarity_sum = 0.0
    for headline in headlines:
        blob = TextBlob(headline)
        polarity_sum += blob.sentiment.polarity

    avg_polarity = polarity_sum / len(headlines)

    if avg_polarity > 0.1:
        return "Positive", round(avg_polarity, 2)
    elif avg_polarity < -0.1:
        return "Negative", round(avg_polarity, 2)
    else:
        return "Neutral", round(avg_polarity, 2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No ticker symbol provided"}))
        sys.exit(1)

    ticker = sys.argv[1]
    headlines, error = fetch_news(ticker)

    if error:
        print(json.dumps({"error": error}))
        sys.exit(1)

    sentiment, score = analyze_sentiment(headlines)

    result = {
        "ticker": ticker.upper(),
        "sentiment": sentiment,
        "confidence": abs(score),
        "headlines": headlines[:5],
        "summary": f"Recent news suggests a {sentiment.lower()} outlook on {ticker.upper()} with confidence score {abs(score)}."
    }

    print(json.dumps(result))
