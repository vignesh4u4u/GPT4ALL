from transformers import pipeline
from transformers import AutoformerModel,AutoTokenizer,AutoModelForSequenceClassification,AutoModelForTokenClassification
classifier = pipeline("ner")
res = classifier("Vignesh and Apple")
print(res)
model_name = "xlnet-base-cased"
ner_model1 = "dbmdz/bert-large-cased-finetuned-conll03-english"
ner_model2 = "allenai/scibert_scivocab_cased"
ner_model3 = "dslim/bert-base-NER"
ner_model4 = "dbmdz/bert-large-cased-finetuned-ontonotes-ner"
model = AutoModelForTokenClassification.from_pretrained(ner_model2)
tokenizer = AutoTokenizer.from_pretrained(ner_model2)
ner_classifier_custom = pipeline("ner", model=model, tokenizer=tokenizer)
ner_result_custom = ner_classifier_custom("vignesh and Apple")
label_mapping = {
    "LABEL_1": "PERSON",
}
filtered_results = []
for entity in ner_result_custom:
    label = label_mapping.get(entity["entity"])
    filtered_results.append({
        "word": entity["word"]
    })
print(filtered_results)




