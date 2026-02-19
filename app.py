import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import string
from nltk.corpus import stopwords

# Download stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('fake_news_model')
tokenizer = BertTokenizer.from_pretrained('fake_news_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit app
st.title('Fake News Detection Chatbot')
st.write('Enter a news article to check if it is fake or true.')

user_input = st.text_area('News Article:', height=200)

if st.button('Predict'):
    if user_input:
        # Clean input
        cleaned_input = clean_text(user_input)

        # Tokenize input
        encoding = tokenizer.encode_plus(
            cleaned_input,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Display result
        if probabilities[1] > probabilities[0]:
            st.success('This news is likely **true**.')
        else:
            st.error('This news is likely **fake**.')
        st.write(f'Confidence: {max(probabilities):.2%}')
    else:
        st.warning('Please enter a news article.')