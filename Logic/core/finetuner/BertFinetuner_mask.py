import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import evaluate
import numpy as np
from datasets import Dataset
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=top_n_genres).to(self.device)

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, 'r') as file:
            self.data = json.load(file)
        self.df = [
            {'summary': movie['first_page_summary'][0], 'genre': movie['genres'][0]} for movie in self.data.values()
        ]
        print("Dataset loaded successfully.")

    def preprocess_genre_distribution(self, top_n_genres=5):
        """
        Preprocess the dataset by filtering for the top n genres and updating the DataFrame.

        Args:
            top_n_genres (int): The number of top genres to filter.
        """
        genre_counts = Counter([entry['genre'] for entry in self.df])
    
        self.top_genres = [genre for genre, _ in genre_counts.most_common(top_n_genres)]
    
        filtered_df = [entry for entry in self.df if entry['genre'] in self.top_genres]
    
        self.df = filtered_df
    
        df_genres = pd.DataFrame(self.df)
    
        genre_counts_filtered = df_genres['genre'].value_counts()
        genre_counts_filtered.plot(kind='bar')
        plt.show()
    
        print("Preprocessing completed. Top genres:", self.top_genres)


    def split_dataset(self, test_size=0.1, val_size=0.1):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        summaries = [entry['summary'] for entry in self.df]
        genres = [entry['genre'] for entry in self.df]
    
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            summaries, genres, test_size=test_size, stratify=genres
        )
    
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=val_size, stratify=train_labels
        )
    
        self.label2id = {genre: idx for idx, genre in enumerate(self.top_genres)}
        self.id2label = {idx: genre for genre, idx in self.label2id.items()}
    
        train_labels = [self.label2id[label] for label in train_labels]
        val_labels = [self.label2id[label] for label in val_labels]
        test_labels = [self.label2id[label] for label in test_labels]
    
        self.train_texts = train_texts
        self.val_texts = val_texts
        self.test_texts = test_texts
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels

        self._clean_data(self.train_texts, self.train_labels)
        self._clean_data(self.val_texts, self.val_labels)
        self._clean_data(self.test_texts, self.test_labels)
    
        print("Dataset split into train, validation, and test sets.")
        
    def _clean_data(self, texts, labels):
        """
        Remove None or mismatched types from the datasets.
    
        Args:
            texts (list): List of text data.
            labels (list): List of labels corresponding to the texts.
        """
        none_indices = [i for i, text in enumerate(texts) if not isinstance(text, str)]
    
        for i in sorted(none_indices, reverse=True):
            del texts[i]
            del labels[i]

    def create_dataset(self, texts, labels):
        """
        Create a PyTorch dataset from the given texts and labels.

        Args:
            texts (list): The input texts.
            labels (list): The corresponding labels.

        Returns:
            Dataset: A Hugging Face Dataset object.
        """
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=512)
        dataset = Dataset.from_dict({'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask'], 'labels': labels})
        return dataset

    def fine_tune_bert(self, epochs=5, batch_size=8, warmup_steps=100, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        train_dataset = self.create_dataset(self.train_texts, self.train_labels)
        val_dataset = self.create_dataset(self.val_texts, self.val_labels)
    
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=50
        )

        trainer = Trainer(
            model=self.model.to('cuda' if torch.cuda.is_available() else 'cpu'),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
    
        trainer.train()
        print("Training completed.")

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        test_dataset = self.create_dataset(self.test_texts, self.test_labels)
        trainer = Trainer(model=self.model, compute_metrics=self.compute_metrics)
        results = trainer.evaluate(eval_dataset=test_dataset)
        print("Model evaluation completed. Results:", results)

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)
        print(f"Model saved to {model_name}.")
        
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    