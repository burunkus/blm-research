#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import os
import sys
import torch
import time
import numpy as np
import random
import datetime
import logging
import logging.handlers
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
from offensive_configs import (
    DATA_PATH,
    SAVE_PATH, 
    CUSTOM_NAME,
    NUMBER_OF_LABELS,
    USE_MEAN_REPR,
    hyperparameters,
    parameter_for_sentiment_model,
    parameters_for_offensive_model
)
from fine_tune_offensive_utils import CustomTextDataset, flat_accuracy, format_time, train
sys.path.append('/home/eokpala/blm-research/scripts')
from fine_tuning_module import CustomBertModel, OffensiveNetworkWithOneFullyConnectedLayer

seed_val = 23
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def evaluate(dataloader, model):
    '''
    Classify the sentiment of tweets in dataloader
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
        # Get probabilities and move to CPU
        predictions = nn.functional.softmax(logits, dim=-1)
        predictions = predictions.detach().cpu().numpy()
        predictions = np.argmax(predictions, axis=1).flatten()
        predicted_labels.extend(predictions)
        
        # Move labels to CPU
        labels = batch_labels.to('cpu').numpy()
        labels = labels.flatten()
        gold_labels.extend(labels)
    
    print(classification_report(gold_labels, predicted_labels, digits=4))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
    print(f"\nMacro F1: {macro_f1:.3f}")
    print(f"Macro Precision: {macro_precision:.3f}")
    print(f"Macro Recall: {macro_recall:.3f}")
    print(f"Micro F1: {micro_f1:.3f}")
    print(f"Micro Precision: {micro_precision:.3f}")
    print(f"Micro Recall: {micro_recall:.3f}")


def main():
    path = DATA_PATH
    test_dataset = path + "offensive_dataset_preprocessed_test.csv"
    
    # Path where the pre-trained model and tokenizer can be found
    model_path = "vinai/bertweet-base"

    print("Loading the pretrained HateBERT model ...")
    model = CustomBertModel(model_path, NUMBER_OF_LABELS)
    print("Pretrained HateBERT model loaded.")
    
    trained_offensive_model_path = SAVE_PATH + 'fine_tuned_bertweet/fine_tuned_fine_tuned_bertweet/fine_tuned_fine_tuned_bertweet.pth'
    logging.info("Loading the fine-tuned offensive network weights ...")
    model.load_state_dict(torch.load(trained_offensive_model_path))
    logging.info("Fine-tuned offensive network weights loaded ...")
    
    batch_size, learning_rate, epochs = hyperparameters()
    
    print("Loading and batching datasets ...")
    tokenizer = model.tokenizer
    test_data = CustomTextDataset(tokenizer, test_dataset)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    print("Datasets loaded and batched.")
    
    print("Evaluation started ...")
    evaluate(test_dataloader, model)
    print("Evaluation completed.")
    
    
if __name__ == "__main__":
    main()
