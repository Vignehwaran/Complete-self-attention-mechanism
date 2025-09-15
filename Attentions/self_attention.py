from regex import T
import torch.nn as nn
import torch

import Attentions

#self-attention mechanism with parameters

class self_attention_mach(nn.Module):
    def __init__(self,input_dim,out_dim):
        super().__init__()      
        self.W_query=nn.Parameter(torch.rand(input_dim,out_dim))
        self.W_key=nn.Parameter(torch.rand(input_dim,out_dim))
        self.W_values=nn.Parameter(torch.rand(input_dim,out_dim)) 

    def forword(self,x):
        Q_W=x @ self.W_query
        Q_k=x @ self.W_key
        Q_v=x @ self.W_values     

        attention_score=Q_W @ Q_k.T
        scaling=attention_score / Q_k.shape[-1]**0.5
        attention_weight=torch.softmax(scaling,dim=-1)
        self.summ=attention_weight[0].sum()
        contextual_embedding=attention_weight @ Q_v
        return contextual_embedding   
    


class self_attention_mechv2(nn.Module):
    def __init__(self,input_dim,out_dim,biases=False):
        super(self_attention_mechv2,self).__init__()
        self.w_query=nn.Linear(input_dim,out_dim,bias=biases)
        self.w_key=nn.Linear(input_dim,out_dim,bias=biases)
        self.w_values=nn.Linear(input_dim,out_dim,bias=biases)


    def forword(self,x):
        Q_W=self.w_query(x)
        Q_k=self.w_key(x)
        Q_v=self.w_values(x)
        
        attention_score= Q_W @  Q_k.T
        scaling= attention_score / Q_k.shape[-1]**0.5
        attention_weight= torch.softmax(scaling ,dim=-1)
        contextual=attention_weight @ Q_v

        return contextual

    

