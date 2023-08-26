from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS,Chroma
from langchain.llms import GPT4All,GooglePalm,TextGen
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceBgeEmbeddings
from pdfminer.high_level import extract_text
from langchain.retrievers import TFIDFRetriever
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import pypdfium2
import PyPDF2
import langchain
import openai
import os
loader=PyPDFLoader("Lease (5).pdf")
document=loader.load_and_split()
print(len(document))
#print(document[0].page_content)
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=100)
text_segments = text_splitter.split_documents(document)
print(len(text_segments))
embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
model_path = os.path.abspath("ggml-model-gpt4all-falcon-q4_0.bin")
llm=GPT4All(model=model_path,verbose=None,backend="gptj")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")