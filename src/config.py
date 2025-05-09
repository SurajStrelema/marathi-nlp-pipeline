# Model names and allowed entity types
ner_model_name = "l3cube-pune/marathi-ner"
sentiment_model_name = "l3cube-pune/marathi-sentiment-md"

allowed_types = {"PERSON", "ORGANIZATION", "DESIGNATION", "LOCATION", "DATE"}

# File paths
input_file = "data/DF-test-dataset.xlsx"
output_file = "output/final_ner_sentiment_output.xlsx"
