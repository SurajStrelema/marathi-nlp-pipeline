import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForSequenceClassification
from .config import ner_model_name, sentiment_model_name, allowed_types

# Load NER model
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

# Load sentiment model
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

# Main function to process each comment
def process_comment(comment):
    comment_str = str(comment)
    try:
        ner_results = ner_pipeline(comment_str)
        entities = [e['word'] for e in ner_results if e['entity_group'].upper() in allowed_types]
        entities_str = ", ".join(entities) if entities else ""
    except:
        entities_str = ""
    
    try:
        sentiment_result = sentiment_pipeline(comment_str)[0]
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']
    except:
        sentiment_label = "UNKNOWN"
        sentiment_score = 0.0

    return pd.Series([entities_str, sentiment_label, sentiment_score])
