from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from sklearn.cluster import KMeans
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple


class AdvancedNLPProcessor:
    def __init__(self):
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        self.keyword_model = KeyBERT()

    def get_bert_embeddings(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.sentence_model.encode(text1)
        emb2 = self.sentence_model.encode(text2)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def extract_key_concepts(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        return self.keyword_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        # Implementation using a sentiment analysis model
        return {"positive": 0.7, "negative": 0.2, "neutral": 0.1}  # Placeholder

    def cluster_resources(self, resource_texts: List[str], n_clusters: int = 5) -> Dict[int, List[int]]:
        embeddings = self.sentence_model.encode(resource_texts)
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(embeddings)

        cluster_dict = {}
        for idx, cluster in enumerate(clusters):
            if cluster not in cluster_dict:
                cluster_dict[cluster] = []
            cluster_dict[cluster].append(idx)

        return cluster_dict

    def generate_summary(self, text: str, ratio: float = 0.2) -> str:
        # Implementation using summarization model
        sentences = text.split('.')
        return '.'.join(sentences[:int(len(sentences) * ratio)]) + '.'