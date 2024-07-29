#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import torch
import time
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from torch import nn
from _datetime import datetime as dt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support, ConfusionMatrixDisplay, confusion_matrix
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
    AutoModelForSequenceClassification
)
from sentiment_configs import (
    custom_bert_with_fc_parameters,
    hyperparameters, 
    NUMBER_OF_LABELS,
    DATA_PATH,
    custom_bert_parameters,
    USE_CUSTOM_BERT,
    SAVE_PATH, 
    CUSTOM_NAME
)
from fine_tuning_utils import CustomTextDataset, flat_accuracy, format_time, train
sys.path.append('/home/eokpala/blm-research/scripts')
from fine_tuning_module import CustomBertModel

USE_CUSTOM_BERT = True
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def evaluate(dataloader, model):
    '''
    Classify the sentiment of the sampled tweets in dataloader
    Args:
        dataloader (Iterable): A PyTorch iterable object through a dataset
        tokenizer (Object): BERT tokenizer
        model (Object): The BERT model
        num_labels (Int): The number of classes in our task
        save_as (String): Absolute path where the classified tweets will be saved
    Returns:
        None
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Put the model in evaluation mode
    model.eval()
    
    # Store true lables for global eval
    gold_labels = []
    # Store predicted labels for global eval
    predicted_labels = []
            
    # Evaluate data for one epoch
    for batch in dataloader:
        # Add batch to GPU/CPU
        batch_input_ids = batch[0].to(device)
        batch_input_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)
        batch = (batch_input_ids, batch_input_mask, batch_labels)

        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            outputs, _ = model(batch)
            logits = outputs.logits  

        predictions = nn.functional.softmax(logits, dim=-1)
        # Move probabilities to CPU
        predictions = predictions.detach().cpu().numpy()
        predictions_flattened = np.argmax(predictions, axis=1).flatten()

        # Move labels to CPU
        labels = batch_labels.to('cpu').numpy()
        labels = labels.flatten()
        
        # Store gold labels single list
        gold_labels.extend(predictions_flattened)
        # Store predicted labels single list
        predicted_labels.extend(labels)
            
    print(classification_report(gold_labels, predicted_labels, digits=4))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
    print(f"\nMacro F1: {macro_f1:.3f}")
    print(f"Macro Precision: {macro_precision:.3f}")
    print(f"Macro Recall: {macro_recall:.3f}")
    print(f"Micro F1: {micro_f1:.3f}")
    print(f"Micro Precision: {micro_precision:.3f}")
    print(f"Micro Recall: {micro_recall:.3f}")
    

if __name__ == "__main__":
    test_dataset = DATA_PATH + "sentiment140_test.txt"
    
    model_path = "vinai/bertweet-base"
    tokenizer_path = "vinai/bertweet-base"
    
    batch_size, learning_rate, epochs = hyperparameters()
    num_labels = NUMBER_OF_LABELS
    
    classifier_criterion = custom_bert_parameters()
    classifier = CustomBertModel(model_path,
                                 num_labels)
    
    tokenizer = classifier.tokenizer
    test_data = CustomTextDataset(tokenizer, test_dataset)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    model_path = SAVE_PATH + "bertweet/fine_tuned_bertweet/fine_tuned_bertweet.pth"
    classifier.load_state_dict(torch.load(model_path))
    evaluate(test_dataloader, classifier)
