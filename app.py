import streamlit as st
import pandas as pd
import sentiment_analysis
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re

# Initialize the  analyzer
analyzer = sentiment_analysis.SentimentIntensityAnalyzer()

def preprocess_email_text(text):
    """
    Comprehensive email preprocessing to remove headers, signatures, URLs, etc.
    
    Args:
        text (str): Raw email text
        
    Returns:
        tuple: (cleaned_text, preprocessing_summary)
    """
    original_length = len(text)
    preprocessing_steps = []
    
    # Convert to string if not already
    text = str(text)
    
    # 1. Remove email headers and reply indicators
    reply_patterns = [
        r'From:.*?\n',
        r'To:.*?\n', 
        r'Cc:.*?\n',
        r'Bcc:.*?\n',
        r'Subject:.*?\n',
        r'Date:.*?\n',
        r'Sent:.*?\n',
        r'On.*?wrote:.*?\n',
        r'On.*?at.*?wrote:.*?\n',
        r'-----Original Message-----.*?(?=\n[A-Z]|\n\n|$)',
        r'________________________________.*?(?=\n[A-Z]|\n\n|$)',
        r'>.*?\n',
        r'^\s*>+.*$'
    ]
    
    for pattern in reply_patterns:
        matches = len(re.findall(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE))
        if matches > 0:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
            preprocessing_steps.append(f"Removed {matches} reply/header elements")
    
    # 2. Remove email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_matches = len(re.findall(email_pattern, text))
    if email_matches > 0:
        text = re.sub(email_pattern, '[EMAIL_REMOVED]', text)
        preprocessing_steps.append(f"Removed {email_matches} email addresses")
    
    # 3. Remove URLs
    url_patterns = [
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?:/[^\s]*)?'
    ]
    
    url_count = 0
    for pattern in url_patterns:
        matches = len(re.findall(pattern, text))
        if matches > 0:
            text = re.sub(pattern, '[URL_REMOVED]', text)
            url_count += matches
    
    if url_count > 0:
        preprocessing_steps.append(f"Removed {url_count} URLs")
    
    # 4. Remove common signature phrases and disclaimers
    signature_patterns = [
        r'best regards.*?(?=\n\n|\n[A-Z]|$)',
        r'kind regards.*?(?=\n\n|\n[A-Z]|$)',
        r'sincerely.*?(?=\n\n|\n[A-Z]|$)',
        r'yours truly.*?(?=\n\n|\n[A-Z]|$)',
        r'thank you.*?(?=\n\n|\n[A-Z]|$)',
        r'thanks.*?(?=\n\n|\n[A-Z]|$)',
        r'cheers.*?(?=\n\n|\n[A-Z]|$)',
        r'sent from my.*?(?=\n\n|\n[A-Z]|$)',
        r'confidential.*?(?=\n\n|\n[A-Z]|$)',
        r'disclaimer.*?(?=\n\n|\n[A-Z]|$)',
        r'this email.*?confidential.*?(?=\n\n|\n[A-Z]|$)',
        r'please consider.*?environment.*?(?=\n\n|\n[A-Z]|$)',
        r'think before.*?print.*?(?=\n\n|\n[A-Z]|$)',
        r'save paper.*?(?=\n\n|\n[A-Z]|$)',
        r'virus.*?free.*?(?=\n\n|\n[A-Z]|$)',
        r'unsubscribe.*?(?=\n\n|\n[A-Z]|$)',
        r'do not reply.*?(?=\n\n|\n[A-Z]|$)'
    ]
    
    signature_count = 0
    for pattern in signature_patterns:
        matches = len(re.findall(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE))
        if matches > 0:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
            signature_count += matches
    
    if signature_count > 0:
        preprocessing_steps.append(f"Removed {signature_count} signature/disclaimer elements")
    
    # 5. Remove environmental footers
    env_patterns = [
        r'please consider the environment.*?(?=\n\n|\n[A-Z]|$)',
        r'think green.*?(?=\n\n|\n[A-Z]|$)',
        r'save the planet.*?(?=\n\n|\n[A-Z]|$)',
        r'go green.*?(?=\n\n|\n[A-Z]|$)'
    ]
    
    env_count = 0
    for pattern in env_patterns:
        matches = len(re.findall(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE))
        if matches > 0:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
            env_count += matches
    
    if env_count > 0:
        preprocessing_steps.append(f"Removed {env_count} environmental footers")
    
    # 6. Remove phone numbers
    phone_patterns = [
        r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        r'\(\d{3}\)\s?\d{3}[-.]?\d{4}'
    ]
    
    phone_count = 0
    for pattern in phone_patterns:
        matches = len(re.findall(pattern, text))
        if matches > 0:
            text = re.sub(pattern, '[PHONE_REMOVED]', text)
            phone_count += matches
    
    if phone_count > 0:
        preprocessing_steps.append(f"Removed {phone_count} phone numbers")
    
    # 7. Clean up extra whitespace and empty lines
    original_lines = len(text.split('\n'))
    
    # Remove multiple consecutive spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    # Remove empty lines at start and end
    text = text.strip()
    
    # Remove lines that are just whitespace or special characters
    text = '\n'.join(line for line in text.split('\n') 
                    if line.strip() and not re.match(r'^[^\w\s]*$', line.strip()))
    
    cleaned_lines = len(text.split('\n'))
    if original_lines != cleaned_lines:
        preprocessing_steps.append(f"Cleaned whitespace ({original_lines} → {cleaned_lines} lines)")
    
    # Create preprocessing summary
    final_length = len(text)
    reduction_percentage = ((original_length - final_length) / original_length) * 100 if original_length > 0 else 0
    
    summary = {
        'original_length': original_length,
        'final_length': final_length,
        'reduction_percentage': reduction_percentage,
        'steps_performed': preprocessing_steps
    }
    
    return text, summary

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

def get_sentiment_color(sentiment):
    """Return color based on sentiment"""
    colors = {
        'positive': '#28a745',
        'neutral': '#ffc107', 
        'negative': '#dc3545'
    }
    return colors.get(sentiment.lower(), '#6c757d')

def get_sentiment_emoji(sentiment):
    """Return emoji based on sentiment"""
    emojis = {
        'positive': '😊',
        'neutral': '😐',
        'negative': '😞'
    }
    return emojis.get(sentiment.lower(), '🤔')

def create_sentiment_gauge(compound_score):
    """Create a gauge chart for sentiment visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = compound_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Score"},
        delta = {'reference': 0},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.05], 'color': "lightcoral"},
                {'range': [-0.05, 0.05], 'color': "lightyellow"},
                {'range': [0.05, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': compound_score
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="Email Sentiment Analysis",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .sentiment-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 8px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #666;
        border-top: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<h1 class="main-header">📧 Email Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze the emotional tone of your emails with advanced sentiment analysis</p>', unsafe_allow_html=True)
    

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Email Content")
        
        # Email input textarea
        user_input = st.text_area(
            "Paste your email content below:",
            height=300,
            placeholder="Dear [Name],\n\nI hope this email finds you well. I wanted to reach out regarding...\n\nBest regards,\n[Your Name]",
            help="Paste the email text you want to analyze for sentiment"
        )
        
        # Analysis button
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if user_input.strip():
                # Perform preprocessing
                with st.spinner("Preprocessing email text..."):
                    cleaned_text, preprocessing_summary = preprocess_email_text(user_input)
                
                # Show preprocessing results
                if preprocessing_summary['steps_performed']:
                    with st.expander("📋 Preprocessing Summary", expanded=False):
                        st.write(f"**Original text length:** {preprocessing_summary['original_length']} characters")
                        st.write(f"**Cleaned text length:** {preprocessing_summary['final_length']} characters")
                        st.write(f"**Reduction:** {preprocessing_summary['reduction_percentage']:.1f}%")
                        
                        st.write("**Steps performed:**")
                        for step in preprocessing_summary['steps_performed']:
                            st.write(f"• {step}")
                        
                        if st.checkbox("Show cleaned text"):
                            st.text_area("Cleaned Text:", value=cleaned_text, height=150, disabled=True)
                
                # Perform sentiment analysis on cleaned text
                if cleaned_text.strip():
                    with st.spinner("Analyzing sentiment..."):
                        vs = analyzer.polarity_scores(cleaned_text)
                        sentiment = classify_sentiment(vs['compound'])
                        
                        # Store results in session state
                        st.session_state.analysis_results = {
                            'sentiment': sentiment,
                            'scores': vs,
                            'timestamp': datetime.now(),
                            'original_length': len(user_input),
                            'cleaned_length': len(cleaned_text),
                            'original_word_count': len(user_input.split()),
                            'cleaned_word_count': len(cleaned_text.split()),
                            'preprocessing_summary': preprocessing_summary,
                            'cleaned_text': cleaned_text
                        }
                else:
                    st.warning("After preprocessing, no meaningful content remains for analysis. Please check your input.")
            else:
                st.error("Please enter some email content to analyze.")

    with col2:
        with st.expander('About This Tool:', expanded=True):
            st.markdown("### ℹ️ About This Tool")
            st.markdown("**Developed By Sudhakar G**")
            st.info("""
            **How it works:**
            Sentiment analysis algorithm breaks down the text into individual words and phrases, assigning each a sentiment score ranging from -1 (negative) to 1 (positive). It then calculates an overall sentiment score for the entire text.

            🔸 **Sentiment Analysis**: Uses lexicon and rule-based approach

            🔸 **Smart Preprocessing**: Removes headers, signatures, URLs, disclaimers
            
            🔸 **Scoring System**: 
            - Positive: ≥ 0.05
            - Neutral: -0.05 to 0.05  
            - Negative: ≤ -0.05
            
            🔸 **Perfect for**: Customer emails, feedback, communication analysis
            """)
        with st.expander('Features:', expanded=False):
            st.markdown("### 📊 Features")
            st.success("""
            ✅ Real-time analysis
            
            ✅ Smart text preprocessing
            
            ✅ Visual sentiment gauge
            
            ✅ Detailed score breakdown
            
            ✅ Professional insights
            """)

    # Display results if analysis has been performed
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        sentiment = results['sentiment']
        vs = results['scores']
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Main sentiment result
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            sentiment_color = get_sentiment_color(sentiment)
            sentiment_emoji = get_sentiment_emoji(sentiment)
            
            st.markdown(f"""
            <div class="sentiment-card" style="background-color: {sentiment_color}20; border: 2px solid {sentiment_color};">
                <h2 style="color: {sentiment_color}; margin: 0;">
                    {sentiment_emoji} {sentiment.upper()} SENTIMENT
                </h2>
                <h3 style="color: {sentiment_color}; margin: 0.5rem 0;">
                    Score: {vs['compound']:.3f}
                </h3>
            </div>
            """, unsafe_allow_html=True)

        # Detailed metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Detailed Scores")
            
            # Create metrics
            st.metric("Positive", f"{vs['pos']:.3f}", delta=None)
            st.metric("Neutral", f"{vs['neu']:.3f}", delta=None)
            st.metric("Negative", f"{vs['neg']:.3f}", delta=None)
            st.metric("Compound", f"{vs['compound']:.3f}", delta=None)
        
        with col2:
            st.markdown("### 📊 Sentiment Gauge")
            fig = create_sentiment_gauge(vs['compound'])
            st.plotly_chart(fig, use_container_width=True)

        # Additional insights
        st.markdown("### 💡 Analysis Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Length", f"{results['original_length']} chars")
            
        with col2:
            st.metric("Cleaned Length", f"{results['cleaned_length']} chars")
            
        with col3:
            reduction = ((results['original_length'] - results['cleaned_length']) / results['original_length'] * 100) if results['original_length'] > 0 else 0
            st.metric("Text Reduced", f"{reduction:.1f}%")

        # Word count metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Words", f"{results['original_word_count']}")
            
        with col2:
            st.metric("Cleaned Words", f"{results['cleaned_word_count']}")
            
        with col3:
            confidence = abs(vs['compound'])
            confidence_level = "High" if confidence > 0.5 else "Medium" if confidence > 0.1 else "Low"
            st.metric("Confidence", confidence_level)

        # Preprocessing details
        if results['preprocessing_summary']['steps_performed']:
            with st.expander("🔧 Preprocessing Details"):
                preprocessing_summary = results['preprocessing_summary']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Steps Performed:**")
                    for step in preprocessing_summary['steps_performed']:
                        st.write(f"• {step}")
                
                with col2:
                    st.write("**Text Statistics:**")
                    st.write(f"• Original: {preprocessing_summary['original_length']} characters")
                    st.write(f"• Cleaned: {preprocessing_summary['final_length']} characters")
                    st.write(f"• Reduction: {preprocessing_summary['reduction_percentage']:.1f}%")
                
                if st.button("Show Cleaned Text"):
                    st.text_area("Cleaned Text Used for Analysis:", 
                               value=results['cleaned_text'], 
                               height=200, 
                               disabled=True)

        # Interpretation
        st.markdown("### 🎯 Interpretation")
        
        if sentiment == 'positive':
            st.success(f"""
            **Positive Sentiment Detected** {get_sentiment_emoji('positive')}
            
            This email expresses positive emotions, satisfaction, or enthusiasm. 
            The tone is likely friendly, appreciative, or optimistic.
            """)
        elif sentiment == 'negative':
            st.error(f"""
            **Negative Sentiment Detected** {get_sentiment_emoji('negative')}
            
            This email contains negative emotions, concerns, or dissatisfaction. 
            Consider addressing any issues mentioned with care and empathy.
            """)
        else:
            st.warning(f"""
            **Neutral Sentiment Detected** {get_sentiment_emoji('neutral')}
            
            This email maintains a neutral, factual tone without strong emotional indicators. 
            It's likely professional or informational in nature.
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>📧 Email Sentiment Analysis Tool | Built with Streamlit & Words-weightage  Sentiment Analysis</p>
        <p>🔒 Your data is processed locally and not stored</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
