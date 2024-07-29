import math
import pickle
import os
import torch
import time
import numpy as np
import random
import datetime
import csv
from scipy import stats 
from _datetime import datetime as dt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AdamW,
    get_scheduler,
    AutoTokenizer,
    RobertaTokenizer,
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    BertTokenizer,
    BertTokenizerFast,
    BertConfig,
    RobertaForSequenceClassification,
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoModel
)

seed_val = 23
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

 
class HateBERTModel(nn.Module):
    
    def __init__(self, 
                 model_path, 
                 num_labels):
        super(HateBERTModel, self).__init__()
        self.model_path = model_path
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        

    def forward(self, batch):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_input_ids, batch_input_mask, batch_labels = batch
        outputs = self.bert(batch_input_ids,
                            token_type_ids=None,
                            attention_mask=batch_input_mask,
                            output_hidden_states=True
                            )
        
        last_hidden_state, pooler_output, hidden_states = outputs.last_hidden_state, outputs.pooler_output, outputs.hidden_states
        return last_hidden_state, pooler_output


class BareBertModel(nn.Module):
    def __init__(self, 
                 model_path):
        super(BareBertModel, self).__init__()
        self.model_path = model_path
        self.bert = AutoModel.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        

    def forward(self, batch):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_input_ids, batch_input_mask, batch_labels = batch
        outputs = self.bert(batch_input_ids,
                            token_type_ids=None,
                            attention_mask=batch_input_mask,
                            output_hidden_states=True
                            )
        
        last_hidden_state, pooler_output, hidden_states = outputs.last_hidden_state, outputs.pooler_output, outputs.hidden_states
        return outputs, pooler_output
    
    
class CustomBertModel(nn.Module):
    
    def __init__(self, 
                 model_path, 
                 num_labels):
        super(CustomBertModel, self).__init__()
        self.model_path = model_path
        self.num_labels = num_labels
        self.bert = AutoModelForSequenceClassification.from_pretrained(self.model_path,
                                                                       num_labels=self.num_labels,
                                                                       output_attentions=False,
                                                                       output_hidden_states=True) 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        
    def forward(self, batch):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_input_ids, batch_input_mask, batch_labels = batch
        if "distilbert" not in self.model_path:
            outputs = self.bert(batch_input_ids,
                                token_type_ids=None,
                                attention_mask=batch_input_mask,
                                labels=batch_labels
                                )
        else:
            # distilbert model don't have token_type_ids
            outputs = self.bert(batch_input_ids,
                                attention_mask=batch_input_mask,
                                labels=batch_labels
                                )
        hidden_states = outputs.hidden_states
        last_hidden_layer = hidden_states[-1] 
        cls_representations = last_hidden_layer[:, 0, :] 
        
        return outputs, cls_representations # outputs, CLS repr
    
    
class OffensiveNetworkWithOneFullyConnectedLayer(nn.Module):
    '''Simple network that concatenates the cls of offensive and cls of sentiment,
       passing the joint representation through a single output FC layer + softmax for classification. 
    '''
    
    def __init__(self, 
                 hate_model, 
                 sentiment_model, 
                 num_labels, 
                 use_mean_repr=False, 
                 is_fine_tuning_offensive=True):
        super(OffensiveNetworkWithOneFullyConnectedLayer, self).__init__()
        self.hate_model = hate_model
        self.sentiment_model = sentiment_model
        self.num_labels = num_labels
        self.use_mean_repr = use_mean_repr
        self.is_fine_tuning_offensive = is_fine_tuning_offensive
        self.dense_layer1 = nn.Linear(self.hate_model.bert.config.hidden_size * 2, self.num_labels)
        
        
    def forward(self, input_batch):
        
        # Get the outputs of the hate model
        hate_model_outputs, _ = self.hate_model(input_batch)
        hate_model_hidden_states = hate_model_outputs.hidden_states
        hate_model_last_hidden_state = hate_model_hidden_states[-1]
            
        # Get the outputs of the sentiment model 
        sentiment_model_outputs, _ = self.sentiment_model(input_batch)
        sentiment_model_hidden_states = sentiment_model_outputs.hidden_states
        sentiment_model_last_hidden_state = sentiment_model_hidden_states[-1] 
        
        if self.use_mean_repr:
            hate_model_last_layer_sequence_representations = hate_model_last_hidden_state[:, 1:-1, :] 
            sentiment_model_last_layer_sequence_representations = sentiment_model_last_hidden_state[:, 1:-1, :]
            
            avg_of_hate_model_last_layer_sequence_representations = torch.mean(hate_model_last_layer_sequence_representations, 1)
            avg_of_sentiment_model_last_layer_sequence_representations = torch.mean(sentiment_model_last_layer_sequence_representations, 1)
            
            joint_representation = torch.cat((avg_of_hate_model_last_layer_sequence_representations, avg_of_sentiment_model_last_layer_sequence_representations), -1)
        else:
            # Get the CLS representation from the last layer of the hate and sentiment models
            hate_model_last_hidden_state_cls = hate_model_last_hidden_state[:, 0, :]
            sentiment_model_last_hidden_state_cls = sentiment_model_last_hidden_state[:, 0, :]
            
            # Concatenate CLS representation of input from hate model and sentiment model
            joint_representation = torch.cat((hate_model_last_hidden_state_cls, sentiment_model_last_hidden_state_cls), -1)
        
        logits = self.dense_layer1(joint_representation)
        output = F.softmax(logits, dim=-1)
        return output, logits
    

class OffensiveNetworkWithDeepAttention2(nn.Module):
    
    def __init__(self, 
                 hate_model, 
                 sentiment_model, 
                 num_labels, 
                 activation,
                 hidden1,
                 hidden2,
                 hidden3,
                 hidden4,
                 is_multi_label_classification=False):
        super(OffensiveNetworkWithDeepAttention2, self).__init__()
        self.hate_model = hate_model
        self.sentiment_model = sentiment_model
        self.num_labels = num_labels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.activation = activation
        self.is_multi_label_classification = is_multi_label_classification
        self.dense_layer1 = nn.Linear(self.hate_model.bert.config.hidden_size * 2, self.hidden1)
        self.dense_layer2 = nn.Linear(self.hidden1, self.hidden2) 
        self.dense_layer3 = nn.Linear(self.hidden2, self.hidden3) 
        self.dense_layer4 = nn.Linear(self.hidden3, self.hidden4)
        self.dense_layer5 = nn.Linear(self.hidden4, 1) 
        self.dense_layer6 = nn.Linear(self.hate_model.bert.config.hidden_size, self.num_labels)
        
        
    def forward(self, input_batch):
        
        # Get the outputs of the hate model
        hate_model_outputs, _ = self.hate_model(input_batch)
        hate_model_hidden_states = hate_model_outputs.hidden_states
        hate_model_last_hidden_state = hate_model_hidden_states[-1]
        
        # Get the outputs of the sentiment model
        sentiment_model_outputs, _ = self.sentiment_model(input_batch)
        sentiment_model_hidden_states = sentiment_model_outputs.hidden_states
        sentiment_model_last_hidden_state = sentiment_model_hidden_states[-1]
        
        # Get the words in the last encoder layer 
        hate_model_last_layer_sequence_representations = hate_model_last_hidden_state[:, 1:, :]
        sentiment_model_last_layer_sequence_representations = sentiment_model_last_hidden_state[:, 1:, :]

        # Concatenate representation of words from hate model with representation of words from sentiment model
        joint_representation = torch.cat((hate_model_last_layer_sequence_representations, sentiment_model_last_layer_sequence_representations), -1)
        
        # Compute the intermediate energies (z)
        hidden_layer1 = self.activation(self.dense_layer1(joint_representation))
        hidden_layer2 = self.activation(self.dense_layer2(hidden_layer1))
        hidden_layer3 = self.activation(self.dense_layer3(hidden_layer2))
        hidden_layer4 = self.activation(self.dense_layer4(hidden_layer3))
        
        # Compute the energies 
        energies = self.dense_layer5(hidden_layer4)
        
        # Compute the attention weights
        alphas = F.softmax(energies, dim=1) 
        
        # Compute context vector (v)
        context_vectors = torch.matmul(alphas.reshape((alphas.shape[0], alphas.shape[2], alphas.shape[1])), hate_model_last_layer_sequence_representations) 
        logits = self.dense_layer6(context_vectors)
        
        if self.is_multi_label_classification:
            output = torch.sigmoid(logits).round()
        else:
            output = F.softmax(logits, dim=-1)
        
        return output.squeeze(), logits.squeeze()
