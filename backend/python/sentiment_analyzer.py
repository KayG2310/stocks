import requests

API_KEY = "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"  # Replace with your OpenRouter API key

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def build_prompt(news,company):
    return f"""###NEWS ARTICLES: {news},
               ###COMPANY NAME: {company}"""

data = {
    "model": "mistralai/mistral-7b-instruct",
    "messages": [
        {
            "role": "system",
            "content": """You are a highly experienced financial news analyst specializing in stock market sentiment. Given a news headline or article and a company name, you must analyze it strictly from the perspective of its likely impact on stock prices of that company.
                        Return only:
                        1. Sentiment — one of: Positive, Negative, or Neutral
                        2. Confidence — a number between 0.0 and 1.0 (to 2 decimal places), representing how confident you are in your sentiment classification

                        Your response must be in the exact format:

                        Sentiment: <Positive|Negative|Neutral>  
                        Confidence: <value between 0.0 and 1.0>

                        Do not provide any explanation, reasoning, or additional text.
                        """
        },
        {
            "role": "user",
            "content": build_prompt(news, company)
        }
    ]
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
print(response.json()["choices"][0]["message"]["content"])
