from transformers import pipeline
from transformers import AutoformerModel,AutoTokenizer,AutoModelForSequenceClassification
classifier = pipeline("sentiment-analysis")
res = classifier("i going in trichy. that is bad place")
print(res)
model_name="distilbert-base-uncased-finetuned-sst-2-english"
model=AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier1 = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
res1 = classifier1("i going in trichy. that is bad place")
print(res1)