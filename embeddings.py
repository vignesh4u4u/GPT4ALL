from langchain.embeddings import HuggingFaceEmbeddings
models = HuggingFaceEmbeddings.list_models()
for model in models:
    print(model)