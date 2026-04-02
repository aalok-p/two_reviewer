import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RewardCalculator:
    RUBRIC_WEIGHT =0.6
    LENGTH_WEIGHT =0.2
    FORMAT_WEIGHT =0.2
    
    def __init__(self, model_name="google/flan-t5-small"):
        self.device ="cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer= AutoTokenizer.from_pretrained(model_name)
        self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.hf_model.to(self.device)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)