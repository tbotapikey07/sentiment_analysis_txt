import pandas as pd
import argparse
import sys
import os
from sentiment_analysis import SentimentIntensityAnalyzer


def classify_sentiment(compound_score):
    """
    Classify sentiment based on compound score.
    
    Args:
        compound_score (float): The compound sentiment score
        
    Returns:
        str: 'positive', 'neutral', or 'negative'
    """
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def analyze_csv_sentiment(input_file, output_file, text_column='email_text'):
    """
    Analyze sentiment of text in a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        text_column (str): Name of the column containing text to analyze
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Read CSV file
    print(f"Reading CSV file: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Check if text column exists
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in CSV.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"Total rows to process: {len(df)}")
    print(f"Text column: {text_column}")
    
    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Initialize new columns
    df['sentiment_compound'] = 0.0
    df['sentiment_positive'] = 0.0
    df['sentiment_neutral'] = 0.0
    df['sentiment_negative'] = 0.0
    df['sentiment_label'] = ''
    
    # Process each row
    print("Processing sentiment analysis...")
    for idx, row in df.iterrows():
        text = str(row[text_column])
        
        # Get sentiment scores
        scores = analyzer.polarity_scores(text)
        
        # Store scores
        df.at[idx, 'sentiment_compound'] = scores['compound']
        df.at[idx, 'sentiment_positive'] = scores['pos']
        df.at[idx, 'sentiment_neutral'] = scores['neu']
        df.at[idx, 'sentiment_negative'] = scores['neg']
        df.at[idx, 'sentiment_label'] = classify_sentiment(scores['compound'])
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows...")
    
    print(f"Completed processing {len(df)} rows.")
    
    # Save to output CSV
    print(f"Saving results to: {output_file}")
    try:
        df.to_csv(output_file, index=False)
        print("Successfully saved output file.")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
    
    # Print summary statistics
    print("\n=== Sentiment Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"Positive: {len(df[df['sentiment_label'] == 'positive'])}")
    print(f"Negative: {len(df[df['sentiment_label'] == 'negative'])}")
    print(f"Neutral: {len(df[df['sentiment_label'] == 'neutral'])}")
    print(f"Average compound score: {df['sentiment_compound'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze sentiment of text in a CSV file using VADER sentiment analysis.'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--column', '-c',
        default='email_text',
        help='Name of the column containing text to analyze (default: email_text)'
    )
    
    args = parser.parse_args()
    
    analyze_csv_sentiment(args.input, args.output, args.column)


if __name__ == '__main__':
    main()
