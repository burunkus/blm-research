import pickle
import os
import sys
import csv
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
from sklearn.metrics import roc_auc_score
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
    hyperparameters
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
        model (Object): The BERT based model
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
    # Store probabilities for AUROC performance metric calculation
    probabilities = []

    # Evaluate data for one epoch
    for batch in dataloader:
        # Add batch to GPU/CPU
        batch_input_ids = batch[0].to(device)
        batch_input_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)
        batch = (batch_input_ids, batch_input_mask, batch_labels)

        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the output predictions after softmax/sigmoid and logits before softmax/sigmoid
            outputs, logits = model(batch)
            
        # Get probabilities
        outputs = outputs.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()

        predictions = np.argmax(outputs, axis=1).flatten()
        predicted_labels.extend(predictions)
        
        probabilities.extend(outputs)
        
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
    
    # AUROC score
    # Use the class probabilities and not the predicted label
    # Note: From (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) - In the binary case, the probability estimates correspond to the probability of the class with the greater label, i.e. estimator.classes_[1]
    greater_label_probabilities = []
    for prob in probabilities:
        greater_label_probabilities.append(prob[1])
    roc_output_value = roc_auc_score(gold_labels, greater_label_probabilities)
    print(f'AUROC score: {roc_output_value}')
        
    
def main(hate_model_path, sentiment_model_path, trained_offensive_model_path, batch_size, data, num_labels):
    '''classify tweets into offensive or non-offensive classes'''
    
    logging.info("Constructing the pretrained offensive model graph ...")
    hate_model = CustomBertModel(hate_model_path, num_labels)
    logging.info("Pretrained offensive model graph constructed.")
    
    logging.info("Constructing the sentiment model graph ...")
    sentiment_model = CustomBertModel(sentiment_model_path, num_labels)
    logging.info("Sentiment model graph constructed.")
    
    logging.info("Constructing OffensiveNetwork graph ...")
    model = OffensiveNetworkWithOneFullyConnectedLayer(hate_model, sentiment_model, num_labels, use_mean_repr=False, is_fine_tuning_offensive=False)
    logging.info("OffensiveNetwork graph constructed ...")
    tokenizer = hate_model.tokenizer

    logging.info("Loading the fine-tuned offensive network weights ...")
    model.load_state_dict(torch.load(trained_offensive_model_path))
    logging.info("Fine-tuned offensive network weights loaded ...")
    
    logging.info("Loading dataset ...")
    data_instance = CustomTextDataset(tokenizer, data)
    dataloader = DataLoader(data_instance, batch_size=batch_size, shuffle=True)
    logging.info("Dataset loaded.")
    
    logging.info("Evaluation started ...")
    evaluate(dataloader, model)
    logging.info("Evaluation completed.")


if __name__ == "__main__":
    
    batch_size = 16
    num_labels = NUMBER_OF_LABELS
    
    hate_model_path = "vinai/bertweet-base"
    sentiment_model_path = "vinai/bertweet-base"
    trained_offensive_model_path = SAVE_PATH + "fine_tuned_bertweet_with_one_fc_layer_mean_repr_with_batch_size16_epochs5_max_len128/fine_tuned_fine_tuned_bertweet_with_one_fc_layer_mean_repr_with_batch_size16_epochs5_max_len128/fine_tuned_fine_tuned_bertweet_with_one_fc_layer_mean_repr_with_batch_size16_epochs5_max_len128.pth"
    test_dataset = DATA_PATH + "offensive_dataset_preprocessed_test.csv"
    main(hate_model_path, sentiment_model_path, trained_offensive_model_path, batch_size, test_dataset, num_labels)
    
    # Evaluate 80:20 split
    #trained_offensive_model_path = SAVE_PATH + "fine_tuned_bertweet_with_one_fc_layer_split_data_80_20mean_repr_false_with_batch_size16_epochs5_max_len128_is_fine_tuning_offensive_false/fine_tuned_fine_tuned_bertweet_with_one_fc_layer_split_data_80_20mean_repr_false_with_batch_size16_epochs5_max_len128_is_fine_tuning_offensive_false/fine_tuned_fine_tuned_bertweet_with_one_fc_layer_split_data_80_20mean_repr_false_with_batch_size16_epochs5_max_len128_is_fine_tuning_offensive_false.pth"
    #test_dataset = DATA_PATH + "offensive_dataset_preprocessed_test_80_20_split.csv"
    #main(hate_model_path, sentiment_model_path, trained_offensive_model_path, batch_size, test_dataset, num_labels)
    
    