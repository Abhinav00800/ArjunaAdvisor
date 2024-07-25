import fitz  
import re

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document[page_num]
        text += page.get_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', '', text)
    text = re.sub(r'\d', '', text)
    return text

if __name__ == "__main__":
    raw_text = extract_text_from_pdf("lekl101.pdf")
    cleaned_text = clean_text(raw_text)

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [lemmatizer.lemmatize(word.lower(), pos='v') for word in words if word.lower() not in stop_words and word not in string.punctuation]
        processed_sentences.append(' '.join(words))
    
    return processed_sentences

if __name__ == "__main__":
    processed_text = preprocess_text(cleaned_text)
    

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,BertForQuestionAnswering
from sklearn.model_selection import train_test_split

class ResearchPaperDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model(texts, labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True)
    labels = torch.tensor(labels)

    dataset = ResearchPaperDataset(encodings, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')

if __name__ == "__main__":
    labels = [0, 1]
    train_model(processed_text, labels)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def answer_question(question, text):
    inputs = tokenizer.encode_plus(question, text, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    outputs = model(input_ids, attention_mask=attention_mask)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    answer = tokenizer.convert_tokens_to_ids(input_ids[0][start_index:end_index])
    return tokenizer.decode(answer)