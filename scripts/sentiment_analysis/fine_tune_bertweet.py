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
    PATH,
    custom_bert_parameters,
    USE_CUSTOM_BERT,
    SAVE_PATH, 
    CUSTOM_NAME
)
from fine_tuning_utils import CustomTextDataset, flat_accuracy, format_time, train
sys.path.append('/home/eokpala/blm-research/scripts')
from fine_tuning_module import CustomBertModel


def main():
    path = PATH
    train_dataset = path + "sentiment140_train.txt"
    test_dataset = path + "sentiment140_test.txt"

    # Path where the pre-trained model and tokenizer can be found
    model_path = "vinai/bertweet-base"
    tokenizer_path = "vinai/bertweet-base" 
    current_file_name = os.path.basename(__file__).split('.')[0]
    current_file_name = current_file_name.split('_')
    base_name = current_file_name[2]
    name = base_name + CUSTOM_NAME
    save_path = SAVE_PATH + f"{base_name}/fine_tuned_{name}/"

    batch_size, learning_rate, epochs = hyperparameters()
    num_labels = NUMBER_OF_LABELS

    classifier_criterion = custom_bert_parameters()
    classifier = CustomBertModel(model_path,
                                 num_labels)
    
    tokenizer = classifier.tokenizer
    training_data = CustomTextDataset(tokenizer, train_dataset)
    test_data = CustomTextDataset(tokenizer, test_dataset)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train(
        train_dataloader,
        test_dataloader,
        tokenizer,
        classifier,
        num_labels,
        learning_rate,
        epochs,
        save_path,
        classifier_criterion,
        USE_CUSTOM_BERT
    )

if __name__ == "__main__":
    main()
