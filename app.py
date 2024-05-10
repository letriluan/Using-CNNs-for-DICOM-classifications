import streamlit as st
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
import neuralcoref
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


df = pd.read_csv('/Users/letriluan/Downloads/NLP/cleaned_data.csv', encoding="latin1")

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

def clean_text1(text):
    tokenizer = RegexpTokenizer(r'\b\w+\b')
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    doc = nlp(" ".join(filtered_tokens))
    lemmatized_tokens = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc]

    return " ".join(lemmatized_tokens)

df_new = df.copy()
df_new["article"] = df_new["article"].apply(clean_text1)

"""**Coreference Resolution utility**"""

def resolve_coreferences(text):
    doc = nlp(text)
    if doc._.has_coref:
        return doc._.coref_resolved
    return text

"""**Text matching utility**"""

from sentence_transformers import SentenceTransformer, util

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

"""**Test utility**"""

def user_interaction():
    while True:
        article_id = input("Enter the article ID or type 'quit' to exit: ")
        if article_id.lower() == 'quit':
            break
        print( 'Article ID: ', article_id)
        question = input("Enter your question: ")

        if question.lower() == 'quit':
            break
        print('Your question: ', question)
        try:
            article_id = int(article_id)
            answer = answer_question_from_article(article_id, question, df_new)
            print("Answer:", answer)
        except ValueError:
            print("Invalid article ID. Please enter a numeric ID.")

def main():
    st.title("Question Answering System")

    article_id = st.text_input("Enter the article ID:")
    question = st.text_input("Enter your question:")

    if st.button("Ask"):
        answer = answer_question_from_article(int(article_id), question, df)
        st.write("Answer:", answer)

if __name__ == "__main__":
    main()
