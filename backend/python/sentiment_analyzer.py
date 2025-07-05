import requests
import sys
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='../.env')  # path relative to python file
API_KEY = os.getenv('SENTIMENT')

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def build_prompt(news,company):
    return f"""###NEWS ARTICLES: {news},
               ###COMPANY NAME: {company}"""

def sentiment_analyzer(news, company):
    data = {
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "messages": [
            {
                "role": "system",
                "content": """You are a highly experienced financial news analyst specializing in stock market sentiment. Given a news headline or article and a company name, you must analyze it strictly from the perspective of its likely impact on stock prices of that company.
                            Return only:
                            1. Sentiment — one of: (1,0,-1) (1 for positive, 0 for neutral, -1 for negative) depending upon how the news article is likely to affect the stock price of that company.
                            2. Confidence — a number between 0.0 and 1.0 (to 2 decimal places), representing how much the news article is likely to affect the stock price of the company either positively or negatively or neutral.
                            3. Reasoning — a short explanation of why you chose the sentiment and confidence you did.

                            ###OUTPUT FORMAT (Your response must be in the exact format):
                            Sentiment: <1|0|-1>  
                            Confidence: <value between 0.0 and 1.0>
                            Reasoning: <short explanation of why you chose the sentiment and confidence you did>

                            **Do not provide any explanation, reasoning, or additional text**
                            **Do NOT generate any additional news articles on your own**
                            """
            },
            {
                "role": "user",
                "content": build_prompt(news, company)
            }
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]
    #return response.json()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sentiment_analyzer.py <news> <company>");
        sys.exit(1)
    news = sys.argv[1]
    company = sys.argv[-1]
    try:
        result = sentiment_analyzer(news, company)
        result = result.split("\n")
        print(result[0])
        print(result[1])
    except Exception as e:
        print(e)