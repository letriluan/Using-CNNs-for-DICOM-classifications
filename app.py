import torch
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import BertTokenizer, BertForQuestionAnswering

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',force_download=True)
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased',force_download=True)


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

def main():
    st.title("BERT Question Answering App")

    # Input fields for question and article ID
    question = st.text_input("Enter your question:")
    article_id = st.number_input("Enter the article ID:", value=0, step=1)

    if st.button("Get Answer"):
        # Retrieve article text from a dataframe or database
        article_text = ""  # Replace with your code to retrieve the article text
        if not article_text:
            st.error("Article not found.")
            return

        # Get answer using BERT model
        answer = answer_question_bert(question, article_text)

        # Display the answer
        st.write("Answer:", answer)

if __name__ == "__main__":
    main()
