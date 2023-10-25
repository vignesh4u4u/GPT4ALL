import os
import textwrap
import warnings
warnings.filterwarnings("ignore")
import random
import torch
import uuid

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS,Chroma
from langchain.document_loaders import PDFMinerLoader
from model.machine_learning_models import HuggingFaceKey

class PdfChatbot:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HuggingFaceKey

    def __init__(self):
        self.seed = 42
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def pdfchatconversion(self,file,inputs_query):
        file_path = str(uuid.uuid4()) + ".pdf"
        print("Saving temporary file for processing: ", file_path)
        try:
            file.save(file_path)
            with open(file_path, "rb") as f:
                loader = PDFMinerLoader(file_path)
                documents = loader.load_and_split()
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                print("Removed temporary file that was processed: ", file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings()
        doc = FAISS.from_documents(texts, embedding=embeddings)
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.3, "max_length": 100})
        chain = load_qa_chain(llm, chain_type="stuff", )
        prompts = inputs_query
        pdf_text = doc.similarity_search(prompts)
        result = chain.run(input_documents=pdf_text, question=prompts)

        return result


