import torch
import streamlit as st
from transformers import BertTokenizer, BertForQuestionAnswering
import pandas as pd
import torch.nn.functional as F


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',force_download=True)
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased',force_download=True)

df = pd.read_csv('cleaned_data.csv')

def answer_question_bert(question, context):
    """Function to answer questions using BERT directly from the context."""
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs, return_dict=True)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert the tokens back to the original words
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer


def answer_question_bert(question, context):
    """Function to answer questions using BERT directly from the context, including confidence score."""
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad(): 
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    start_probs = F.softmax(answer_start_scores, dim=-1)
    end_probs = F.softmax(answer_end_scores, dim=-1)
    answer_start = torch.argmax(start_probs)
    answer_end = torch.argmax(end_probs) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer
    
def answer_question_from_article(article_id, question, df):
    """Retrieve an article by ID and use BERT to answer a question based on the article's text, including confidence."""
    try:
        article_text = df.loc[df['id'] == article_id, 'article'].values[0]
    except IndexError:
        return "Article not found."

    answer = answer_question_bert(question, article_text)
    return answer

def main():
    st.title("BERT Question Answering App")

    # Input fields for question and article ID
    question = st.text_input("Enter your question:")
    article_id = st.number_input("Enter the article ID:", value=0, step=1)

    if st.button("Get Answer"):

        # Get answer using BERT model
        answer = answer_question_from_article(article_id, question, df)

        # Display the answer
        st.write("Answer:", answer)

if __name__ == "__main__":
    main()
