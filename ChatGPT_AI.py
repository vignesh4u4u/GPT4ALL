from flask import Flask,Response,request,render_template,json,jsonify
from langchain.llms import GPT4All,OpenAI,OpenAIChat
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from pdfminer.high_level import extract_text
import pypdfium2
import PyPDF2
import langchain
import openai
import os
os.environ["gpt4"]="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"
raw_text=extract_text("Lease (6).pdf")
#print(raw_text)
text_splitter=CharacterTextSplitter(
    separator="\n",chunk_size=700,chunk_overlap=300,length_function=len,
)
text=text_splitter.split_text(raw_text)
#print(len(text))
embedding=OpenAIEmbeddings(openai_api_key=os.environ["gpt4"])
doc_search=FAISS.from_texts(text,embedding)
chain = load_qa_chain(OpenAI(),chain_type="stuff")
query="hello"
docu=doc_search.similarity_search(query)
chain.run(input_documents=docu,question=query)

