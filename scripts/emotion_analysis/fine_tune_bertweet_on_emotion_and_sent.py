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
from _datetime import datetime as dt
from torch import nn
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

CUSTOM_NAME = ""
def get_filename():
    #ct = datetime.datetime.now()
    #log_name = f"{ct.year}-{ct.month:02d}-{ct.day:02d}_{ct.hour:02d}:{ct.minute:02d}:{ct.second:02d}"
    current_file_name = os.path.basename(__file__).split('.')[0]
    log_name = current_file_name
    return log_name


def get_logger(log_folder,log_filename):
    if os.path.exists(log_folder) == False:
        os.makedirs(log_folder)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]:  %(message)s",
        datefmt="%m-%d-%Y %H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(log_folder, log_filename+'.log'), mode='w'),
        logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger


def main():
    path = DATA_PATH
    train_dataset = path + "emotion_preprocessed_train.csv"
    test_dataset = path + "emotion_preprocessed_test.csv"
    
    # Path where the pre-trained model and tokenizer can be found
    sentiment_model_path = "vinai/bertweet-base"
    bertweet_model_path = "vinai/bertweet-base"
    
    current_file_name = os.path.basename(__file__).split('.')[0]
    words_in_file_name = current_file_name.split('_')
    words_in_file_name[1] = "tuned" # Change tune to tuned
    name = "_".join(words_in_file_name) + CUSTOM_NAME
    save_path = SAVE_PATH + f"{name}/{name}/"
    
    sentiment_params = parameter_for_sentiment_model()
    logging.info("Constructing the sentiment model graph ...")
    # Sentiment model graph
    sentiment_model = CustomBertModel(sentiment_model_path, sentiment_params['num_labels'])
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
    
    logging.info("Constructing the bertweet model graph ...")
    # Bertweet model graph
    bertweet_model = BareBertModel(bertweet_model_path)
    logging.info("Bertweet model graph constructed.")
    
    params = parameter_for_emotion_model()
    activation = params['activation'][1]
    hidden_size = params['input_size']
    hidden1 = params['hidden1']
    hidden2 = params['hidden2']
    hidden3 = params['hidden3']
    hidden4 = params['hidden4']
    hidden4 = 256
    logging.info(f"EmotionNetworkWithDeepAttention2 inputs: {params}")
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
    
    
    logging.info("Freezing all layers in the sentiment model to prevent them from being trained ...")
    # Set requires_grad to False to prevent the sentiment model from being trained
    for parameter in model.sentiment_model.bert.parameters():
        parameter.requires_grad = False
    logging.info("Sentiment model of the emotion network frozen.")
    
    batch_size, learning_rate, epochs, patience = hyperparameters_for_early_stopping()
    patience = 7
    num_labels = NUMBER_OF_LABELS
    params = parameters_for_fine_tuning()
    criterion = params['criterion']
    
    logging.info("Loading and batching datasets ...")
    training_data = CustomTextDataset(tokenizer, train_dataset)
    test_data = CustomTextDataset(tokenizer, test_dataset)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    logging.info("Datasets loaded and batched.")
    
    logging.info("Training started ...")
    train(
        train_dataloader,
        test_dataloader,
        tokenizer,
        model,
        num_labels,
        learning_rate,
        epochs,
        save_path,
        criterion,
        logging,
        patience=patience,
        is_layers_frozen=True
    )
    logging.info("Training completed.")
    
    
if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename() + CUSTOM_NAME)
    main()
