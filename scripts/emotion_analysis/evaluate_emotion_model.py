#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import emoji
import json
import datetime
import csv
import sys
import numpy as np
import torch
import pickle
import random
import logging
import logging.handlers
from sklearn.model_selection import train_test_split
from _datetime import datetime as dt
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch import nn
from transformers import (
    AdamW,
    get_scheduler,
    AutoModel,
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    RobertaTokenizer,
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    BertTokenizer,
    BertTokenizerFast,
    BertConfig,
    RobertaForSequenceClassification,
    BertForSequenceClassification
)
from emotion_configs import (
    DATA_PATH,
    SAVE_PATH, 
    CUSTOM_NAME,
    NUMBER_OF_LABELS,
    hyperparameters,
    parameter_for_emotion_model,
    parameter_for_sentiment_model,
    parameters_for_fine_tuning,
    hyperparameters_for_early_stopping
)
from emotion_utils import CustomTextDataset, flat_accuracy, format_time, train
sys.path.append('/home/eokpala/blm-research/scripts')
from fine_tuning_module import CustomBertModel, BareBertModel, OffensiveNetworkWithDeepAttention2 as EmotionNetworkWithDeepAttention2

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
        model (Object): The BERT model
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
            # This will return outputs after sigmoid.round() which will convert the probabilities to binary labels, and logits from the output layer before sigmoid
            outputs, logits = model(batch)
        
        # Get probabilities
        predictions = outputs.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()
        
        # Store gold labels single list
        gold_labels.extend(label_ids)
        # Store predicted labels single list
        predicted_labels.extend(predictions)
            
    print(classification_report(gold_labels, predicted_labels, digits=4))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
    print(f"\nMacro F1: {macro_f1:.3f}")
    print(f"Macro Precision: {macro_precision:.3f}")
    print(f"Macro Recall: {macro_recall:.3f}")
    print(f"Micro F1: {micro_f1:.3f}")
    print(f"Micro Precision: {micro_precision:.3f}")
    print(f"Micro Recall: {micro_recall:.3f}")    
    

def main(base_model_path, trained_emotion_model_path, num_labels, batch_size, data):
    ''' classify tweets into eleven emotion classes'''
    
    sentiment_params = parameter_for_sentiment_model()
    logging.info("Constructing the sentiment model graph ...")
    # Sentiment model graph
    sentiment_model = CustomBertModel(base_model_path, sentiment_params['num_labels'])
    logging.info("Sentiment model graph constructed.")
    
    logging.info("Loading fine-tuned sentiment weights into the constructed sentiment model graph ...")
    # Load the fine-tuned sentiment model
    fine_tuned_sentiment_model_path = "../../../../../../../zfs/socbd/eokpala/blm_research/models/sentiment/bertweet/fine_tuned_bertweet/fine_tuned_bertweet.pth"
    sentiment_model.load_state_dict(torch.load(fine_tuned_sentiment_model_path))
    logging.info("Loaded fine-tuned sentiment weights into the constructed sentiment model graph.")
    tokenizer = sentiment_model.tokenizer
    
    logging.info("Reseting the number of classes in the classification layer of the loaded fine-tuned sentiment model...")
    num_features = sentiment_model.bert.classifier.out_proj.in_features
    sentiment_model.bert.classifier.out_proj = nn.Linear(num_features, NUMBER_OF_LABELS)
    logging.info("Model classification layer reseted with the correct number of classes for the emotion task.")
    
    logging.info("Constructing the emotion model graph ...")
    # Bertweet model graph
    bertweet_model = BareBertModel(base_model_path) 
    logging.info("Emotion model graph constructed.")
    
    params = parameter_for_emotion_model()
    activation = params['activation'][1]
    hidden_size = params['input_size']
    hidden1 = params['hidden1']
    hidden2 = params['hidden2']
    hidden3 = params['hidden3']
    hidden4 = params['hidden4']
    hidden4 = 256

    logging.info("Construct Emotion network graph ...")
    model = EmotionNetworkWithDeepAttention2(sentiment_model, 
                                            bertweet_model,
                                            NUMBER_OF_LABELS,
                                            activation,
                                            hidden1,
                                            hidden2,
                                            hidden3,
                                            hidden4,
                                            is_multi_label_classification=True)
    logging.info("Emotion network graph constructed.")
    
    # Load the fine-tuned emotion model
    logging.info("Loading trained emotion weights into model ...")
    model.load_state_dict(torch.load(trained_emotion_model_path))
    logging.info("Emotion weights loaded.")
    
    logging.info("Loading dataset ...")
    data_instance = CustomTextDataset(tokenizer, data)
    dataloader = DataLoader(data_instance, batch_size=batch_size, shuffle=True)
    logging.info("Dataset loaded.")
    
    logging.info("Evaluation started ...")
    evaluate(dataloader, model)
    logging.info("Evaluation completed.")
    
            
if __name__ == "__main__":
    test_dataset = DATA_PATH + "emotion_preprocessed_test.csv"
    
    batch_size = 8
    num_labels = NUMBER_OF_LABELS
    base_model_path = "vinai/bertweet-base"
    trained_emotion_model_path = SAVE_PATH + "fine_tuned_bertweet_on_emotion_and_sent_mean_repr_with_early_stopping_hidden4_256_patience7_with_max_len128_batch_size8_leaky_relu_ENWDA2/fine_tuned_bertweet_on_emotion_and_sent_mean_repr_with_early_stopping_hidden4_256_patience7_with_max_len128_batch_size8_leaky_relu_ENWDA2/fine_tuned_bertweet_on_emotion_and_sent_mean_repr_with_early_stopping_hidden4_256_patience7_with_max_len128_batch_size8_leaky_relu_ENWDA2.pth"
    main(base_model_path, trained_emotion_model_path, num_labels, batch_size, test_dataset)