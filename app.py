from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re
from nltk.corpus import stopwords
import string

# Download stopwords if not already available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def analyze_sentiment():
    try:
        # Get and clean user input
        text_input = request.form.get('text1', '').lower()
        text_no_digits = re.sub(r'\d+', '', text_input)
        text_clean = ''.join(c for c in text_no_digits if c not in string.punctuation)
        processed_text = ' '.join([word for word in text_clean.split() if word not in stop_words])

        # Perform sentiment analysis
        scores = analyzer.polarity_scores(processed_text)
        compound_score = round((1 + scores['compound']) / 2, 2)  # Normalize to 0-1

        return render_template(
            'form.html',
            final=compound_score,
            text1=text_clean,
            text2=scores['pos'],
            text3=scores['neu'],
            text4=compound_score,
            text5=scores['neg']
        )
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
