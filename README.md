# Amazon Review Analysis

### **Overall Goal**

-   Develop an NLP model to automatically summarize Amazon product reviews and extract key points such as pros, cons, and sentiment.
-   Offer a clear and explainable summary, reducing information overload while maintaining review diversity.
-   Help customers make informed purchasing decisions by providing concise insights from a large volume of reviews.

### **Scope**

-   **Data Collection**: Gather Amazon reviews using pre-existing datasets or scraping methods.
-   **Preprocessing**: Clean, tokenize, and normalize text data.
-   **Summarization**: Implement extractive and/or abstractive summarization models, such as BERT, T5, or TextRank.
-   **Sentiment Analysis**: Use models such as VADER or TextBlob to determine review polarity.
-   **Pros/Cons Extraction**: Apply aspect-based sentiment analysis or keyword extraction to identify product strengths and weaknesses.
-   **Prototype Output**: A web app or extension showcasing summarized reviews, sentiment, and pros/cons.

### **Dataset**

-   https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews

### Setup

pip install pandas as pd
pip install torch spacy
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

### Acknowledgments

This project uses the [Name-Gender-Predictor](https://github.com/imshibl/Name-Gender-Predictor) repository by [Imshibl](https://github.com/imshibl) for predicting gender from names.
