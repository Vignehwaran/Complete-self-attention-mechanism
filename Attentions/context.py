import torch

class ContextualCalculation:
    def __init__(self,word_embeddings):
        self.word_embeddings=word_embeddings

    def attention_score(self):
        return self.word_embeddings @ self.word_embeddings.T 
    
    def attention_weights(self,attention_score):
        return torch.softmax(attention_score,dim=-1) 
     
    def contextualvlaues(self,attention_weights): 
        return attention_weights @ self.word_embeddings  

    