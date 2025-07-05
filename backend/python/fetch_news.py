# fetch_news_newsapi.py

import requests
import sys
import os
import json
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='../.env')  # path relative to python file
API_KEY = os.getenv("NEWSAPI_KEY")

INDIAN_FINANCIAL_DOMAINS = [
    "moneycontrol.com",
    "economictimes.indiatimes.com",
    "business-standard.com",
    "financialexpress.com",
    "cnbctv18.com",
    "zeebiz.com",
    "ndtvprofit.com",
    "iifl.com",
    "bqprime.com"
]

def fetch_indian_stock_news(company_name, limit=5):
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": company_name,
        "language": "en",
        "pageSize": limit,
        "sortBy": "publishedAt",
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code} — {response.text}")
        return []
    articles = response.json().get("articles", [])
    results = []

    for a in articles:
        results.append({
            "title": a["title"],
            "description": a.get("description", ""),
            "url": a["url"],
            "publishedAt": a["publishedAt"]
        })

    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_news_newsapi.py <company_name>")
        sys.exit(1)
    company = " ".join(sys.argv[1:])
    news = fetch_indian_stock_news(company)
    print(json.dumps(news));

    # if not news:
    #     print(f"No news found for '{company}' in Indian finance sources.")
    # else:
    #     print(f"\n📰 Latest News on {company} (India-specific):\n")
    #     for i, item in enumerate(news, 1):
    #         print(f"{i}. {item['title']}")
    #         print(f"   {item['url']}")
    #         print(f"   {item['description']}")
    #         print(f"   Published at: {item['publishedAt']}\n")
