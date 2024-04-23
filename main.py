import streamlit as st
from PyPDF2 import PdfReader
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_conversational_chain():

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                            google_api_key=GOOGLE_API_KEY,
                             temperature=0.3)
    tweet_prompt = PromptTemplate.from_template("Create required number of Jira stories descriptions along with story titles as pointers to develop the application mentioned in the document - {text}.")

    chain = LLMChain(llm=model, prompt=tweet_prompt, verbose=True)

    return chain



def create_stories(raw_text):

    chain = get_conversational_chain()

    
    response = chain.run(text=raw_text)

    return response




def main():
    result = ''
    st.set_page_config("Mock App")
    st.header("Create JIRA stories using Gemini💁")


    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                result=create_stories(raw_text)
                st.success("Done")
    st.write("Jira stories obtained:")
    st.write(result)

if __name__ == "__main__":
    main()