import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import string
import numpy as np
import re

df = pd.read_csv('cleaned_data.csv', encoding="latin1")

nlp = spacy.load('en_core_web_sm')

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def find_most_relevant_sentence(question, article_text):
    doc = nlp(article_text)
    sentences = [sent.text for sent in doc.sents]
    question_embedding = model.encode(question)
    sentence_embeddings = model.encode(sentences)

    similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings).squeeze()
    most_similar_index = similarities.argmax().item()
    confidence = similarities[most_similar_index].item()

    return sentences[most_similar_index], confidence

nlp = spacy.load("en_core_web_sm")
def extract_relevant_snippets(question, relevant_sentence):
    doc = nlp(relevant_sentence)
    question_doc = nlp(question)

    target_label = 'PERSON'  # Default to PERSON

    if any(word in question.lower() for word in ['who', 'name']):
        target_label = 'PERSON'
    elif any(word in question.lower() for word in ['when', 'date', 'year', 'time']):
        target_label = 'DATE'
    elif any(word in question.lower() for word in ['where', 'city', 'country', 'place', 'location']):
        target_label = 'GPE'
    elif any(word in question.lower() for word in ['what', 'company', 'organization']):
        target_label = 'ORG'
    elif 'how many' in question.lower():
        target_label = 'CARDINAL'

    entities = {}
    for ent in doc.ents:
        if ent.label_ == target_label:
            if ent.text in entities:
                entities[ent.text] += 1
            else:
                entities[ent.text] = 1

    if entities:
        sorted_entities = sorted(entities.items(), key=lambda item: (-item[1], relevant_sentence.index(item[0])))
        return sorted_entities[0][0]

    return "No relevant information found."

def answer_question_from_article(article_id, question, df):
    try:
        article_text = df.loc[df['id'] == article_id, 'article'].values[0]
    except IndexError:
        return "Article not found."

    relevant_sentence, confidence = find_most_relevant_sentence(question, article_text)

    confidence_threshold = 0.5
    if confidence < confidence_threshold:
        return "High confidence answer not found."

    answer_snippet = extract_relevant_snippets(question, relevant_sentence)
    return answer_snippet, confidence


def main():
    st.title("BERT Question Answering App")

    # Input fields for question and article ID
    question = st.text_input("Enter your question:")
    article_id = st.number_input("Enter the article ID:", value=0, step=1)

    if st.button("Get Answer"):

        # Get answer using BERT model
        answer = answer_question_from_article(article_id, question, df_new)

        # Display the answer
        st.write("Answer:", answer)

if __name__ == "__main__":
    main()
