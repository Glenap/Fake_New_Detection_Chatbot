# Fake_New_Detection_Chatbot

This project is a simple AI-based chatbot that checks whether a news article or statement is likely to be Real or Fake based on its text content.

The idea behind building this was to explore how NLP and machine learning can be used to tackle misinformation, which is becoming a serious issue across social media and online news platforms.

Live App:
https://fakenewsdetection-9ix5r3mbvwd9vakvdh6j3f.streamlit.app/

## What the app does

Takes news text as input

Processes and cleans the text

Runs it through a trained ML model

Predicts whether it is Fake or Real

Shows the result in a chatbot-style response

The goal was to make the interaction simple — more like asking a question than using a technical tool.

## Tech used

I kept the stack lightweight so it’s easy to run and deploy:

Python

Streamlit (for UI)

Scikit-learn

NLP preprocessing (NLTK / basic text cleaning)

The model is trained using standard text classification techniques like TF-IDF with a supervised classifier.

## How it works (high level)

User enters a news headline or paragraph

Text is preprocessed

Lowercasing

Stopword removal

Tokenization

Text is converted into numerical features

Model predicts Fake or Real

Chatbot displays the response

Running locally

If you want to test it locally:

pip install -r requirements.txt
streamlit run app.py

## Example

Input:
“Scientists confirm humans can live on Mars by 2030.”

Output:
Prediction → Fake News

## Limitations

This is a prototype, so it has some obvious limitations:

Accuracy depends on the dataset used for training

Doesn’t verify sources

Can struggle with satire or opinion content

Not trained on real-time news

## Why I built this

Mainly to experiment with:

NLP pipelines

Text classification

Streamlit deployment

Building a simple AI chatbot interface

It was a good way to combine ML + UI into one small project.


