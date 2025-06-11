from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load model and tokenizer from local directory
tokenizer = AutoTokenizer.from_pretrained("./local_finbert")
model = AutoModelForSequenceClassification.from_pretrained("./local_finbert")

# Create pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test with example texts
sample_texts = "Air India to start flights to Dubai"

result = nlp(sample_texts)[0]
print(f"Sentiment: {result['label']}")
print(f"Confidence: {result['score']:.4f}")
    
