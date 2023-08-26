import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)
model_name = "distilbert-base-cased-distilled-squad"
model_name1 = "bert-large-cased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name1)
model = AutoModelForQuestionAnswering.from_pretrained(model_name1)
passage = "vignesh and yogesh is friend"
question = "vignesh friend is who?"
inputs = tokenizer(question, passage, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits) + 1
answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
print("Question:", question)
print("Predicted Answer:", answer)
