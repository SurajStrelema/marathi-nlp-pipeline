import pandas as pd
from .config import input_file, output_file
from .ner_sentiment import process_comment

def main():
    df = pd.read_excel(input_file)
    df[['Entities', 'Overall_Sentiment', 'Sentiment_Score']] = df['Comments'].apply(process_comment)
    df.to_excel(output_file, index=False)
    print(f"Final output saved to: {output_file}")

if __name__ == "__main__":
    main()
