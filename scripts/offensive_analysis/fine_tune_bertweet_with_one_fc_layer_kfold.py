#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import os
import csv
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
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from scipy import stats
from torch import nn
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
    parameters_of_offensive_model_with_fc_layers,
    parameters_of_offensive_model_with_deep_attention,
    hyperparameters_for_early_stopping,
    parameter_for_sentiment_model,
    parameters_for_offensive_model
)
from fine_tune_offensive_utils import flat_accuracy, format_time, train
sys.path.append('/home/eokpala/blm-research/scripts')
from fine_tuning_module import CustomBertModel, OffensiveNetworkWithOneFullyConnectedLayer


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


class CustomTextDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data,
                 padding="max_length",
                 truncation=True,
                 max_length=128
                 ):

        """
        Generate a single example and its label from data_path
        Args:
            tokenizer (Object): BERT/RoBERTa tokenization object
            data_path (String): Absolute path to the train/test dataset
            padding (String): How to padding sequences, defaults to "max_lenght"
            truncation (Boolean): Whether to truncate sequences, defaults to True
            max_length (Int): The maximum length of a sequence, sequence longer
            than max_length will be truncated and those shorter will be padded
        Retruns:
            dataset_item (Tuple): A tuple of tensors - tokenized text, attention mask and labels
        """

        self.tweet_ids = []
        self.author_ids = []
        self.tweets = []
        self.labels = []
        
        for i, line in enumerate(data):
            self.tweet_ids.append(line[0])
            self.author_ids.append(line[1])
            self.tweets.append(line[2])
            self.labels.append(line[3])

        tokenized_tweet = tokenizer(self.tweets,
                                    padding=padding,
                                    truncation=truncation,
                                    max_length=max_length)

        self.examples = tokenized_tweet['input_ids']
        self.attention_masks = tokenized_tweet['attention_mask']
        self.token_type_ids = tokenized_tweet['token_type_ids']


    def __len__(self):
        return len(self.examples)
    
    
    def __getitem__(self, index):
        dataset_item = (torch.tensor(self.examples[index]),
                        torch.tensor(self.attention_masks[index]),
                        torch.tensor(self.labels[index])
                    )
        return dataset_item
    
    
def evaluate(dataloader, model, with_sentiment=True):
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

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
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
            if with_sentiment:
                # This will return the output predictions after softmax/sigmoid and logits before softmax/sigmoid
                outputs, logits = model(batch)
            else:
                # Only the Custom BERT model
                outputs, _ = model(batch)
                logits = outputs.logits
                outputs = nn.functional.softmax(logits, dim=-1)
            
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
    accuracy = accuracy_score(gold_labels, predicted_labels)
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
    
    return macro_f1, macro_precision, macro_recall, accuracy, roc_output_value
    

def manual_paired_ttest(with_sent_metric_scores, without_sent_metric_scores, cv=10):
    
    score_diff = []
    for i, score in enumerate(with_sent_metric_scores):
        score_diff.append(score - without_sent_metric_scores[i])
    
    avg_diff = np.mean(score_diff)

    numerator = avg_diff * np.sqrt(cv)
    denominator = np.sqrt(
        sum([(diff - avg_diff) ** 2 for diff in score_diff]) / (cv - 1)
    )
    t_stat = numerator / denominator

    pvalue = stats.t.sf(np.abs(t_stat), cv - 1) * 2.0
    return float(t_stat), float(pvalue)
    
    
def main():
    """
    Implements the k-fold paired t test procedure to compare the performance of two models.
    """
    path = DATA_PATH
    dataset_path = path + "offensive_dataset_preprocessed.csv"
    
    dataset = []
    with open(dataset_path, encoding="utf-8-sig") as csv_file_handle:
        csv_reader = csv.reader(csv_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader):
            row = []
            row.append(line[0].strip())
            row.append(line[1].strip())
            row.append(line[2].strip())
            row.append(int(line[3].strip()))
            dataset.append(row)
    
    # With sentiment
    logging.info("Starting with sentiment Kflod cross validation")
    f1_scores_with_sentiment = []
    precision_scores_with_sentiment = []
    recall_scores_with_sentiment = []
    accuracy_scores_with_sentiment = []
    auroc_scores_with_sentiment = []
    kfold = KFold(n_splits=10, random_state=5, shuffle=True)
    for i, (train_index, test_index) in enumerate(kfold.split(dataset)):
        train_dataset, test_dataset = [], []
        logging.info(f'--- Fold {i + 1} ---')
        # Prepare the train set
        for index in train_index:
            train_dataset.append(dataset[index])
        
        # Prepare the test set
        for index in test_index:
            test_dataset.append(dataset[index])
    
        # Path where the pre-trained model and tokenizer can be found
        hate_model_path = "vinai/bertweet-base"
        sentiment_model_path = "vinai/bertweet-base"

        current_file_name = os.path.basename(__file__).split('.')[0]
        words_in_file_name = current_file_name.split('_')
        words_in_file_name[1] = "tuned" # Change tune to tuned
        name = "_".join(words_in_file_name) + CUSTOM_NAME
        save_path = SAVE_PATH + f"{name}/{name}_with_sentiment_{i+1}/"

        logging.info("Loading the pretrained HateBERT model ...")
        hate_model = CustomBertModel(hate_model_path, NUMBER_OF_LABELS)
        logging.info("Pretrained HateBERT model loaded.")

        sentiment_params = parameter_for_sentiment_model()
        logging.info("Constructing the sentiment model graph ...")
        sentiment_model = CustomBertModel(sentiment_model_path, sentiment_params['num_labels'])
        logging.info("Sentiment model graph constructed.")

        logging.info("Loading fine-tuned sentiment weights into the constructed sentiment model graph ...")
        fine_tuned_sentiment_model_path = "../../../../../../../zfs/socbd/eokpala/blm_research/models/sentiment/bertweet/fine_tuned_bertweet/fine_tuned_bertweet.pth"
        sentiment_model.load_state_dict(torch.load(fine_tuned_sentiment_model_path))
        sentiment_tokenizer = sentiment_model.tokenizer
        logging.info("Loaded fine-tuned sentiment weights into the constructed sentiment model graph.")

        logging.info("Constructing OffensiveNetwork graph ...")
        model = OffensiveNetworkWithOneFullyConnectedLayer(hate_model, sentiment_model, NUMBER_OF_LABELS, use_mean_repr=USE_MEAN_REPR)
        logging.info("OffensiveNetwork graph constructed ...")

        logging.info("Freezing all layers in the sentiment model to prevent them from being trained ...")
        # Set requires_grad to False to prevent the sentiment model from being trained
        for parameter in model.sentiment_model.bert.parameters():
            parameter.requires_grad = False
        logging.info("Sentiment model of the OffensiveNetowork frozen.")

        batch_size, learning_rate, epochs = hyperparameters()
        num_labels = NUMBER_OF_LABELS
        criterion = parameters_for_offensive_model()['criterion']

        logging.info("Loading and batching datasets ...")
        tokenizer = hate_model.tokenizer
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
            logging
        )
        logging.info(f"--- Fold {i + 1} training completed ---")
        logging.info(f"Evaluating the performance of fold {i + 1} ...")
        f1_score, precision_score, recall_score, accuracy_score, auroc_score = evaluate(test_dataloader, model)
        logging.info(f"Evaluation of the performance of fold {i + 1} complete")
        f1_scores_with_sentiment.append(f1_score)
        precision_scores_with_sentiment.append(precision_score)
        recall_scores_with_sentiment.append(recall_score)
        accuracy_scores_with_sentiment.append(accuracy_score)
        auroc_scores_with_sentiment.append(auroc_score)
    logging.info("------ Kflod cross validation with sentiment completed ------")
    print()
    
    logging.info("Starting without sentiment Kflod cross validation")
    # Without sentiment
    f1_scores_without_sentiment = []
    precision_scores_without_sentiment = []
    recall_scores_without_sentiment = []
    accuracy_scores_without_sentiment = []
    auroc_scores_without_sentiment = []
    kfold = KFold(n_splits=10, random_state=5, shuffle=True)
    for i, (train_index, test_index) in enumerate(kfold.split(dataset)):
        train_dataset, test_dataset = [], []
        logging.info(f'--- Fold {i + 1} ---')
        # Prepare the train set
        for index in train_index:
            train_dataset.append(dataset[index])
        
        # Prepare the test set
        for index in test_index:
            test_dataset.append(dataset[index])
    
        # Path where the pre-trained model and tokenizer can be found
        model_path = "vinai/bertweet-base"

        current_file_name = os.path.basename(__file__).split('.')[0]
        words_in_file_name = current_file_name.split('_')
        words_in_file_name[1] = "tuned" # Change tune to tuned
        name = "_".join(words_in_file_name) + CUSTOM_NAME
        save_path = SAVE_PATH + f"{name}/{name}_without_sentiment_{i+1}/"

        logging.info("Loading the pretrained HateBERT model ...")
        model = CustomBertModel(model_path, NUMBER_OF_LABELS)
        logging.info("Pretrained HateBERT model loaded.")

        batch_size, learning_rate, epochs = hyperparameters()
        num_labels = NUMBER_OF_LABELS
        criterion = parameters_for_offensive_model()['criterion']

        logging.info("Loading and batching datasets ...")
        tokenizer = hate_model.tokenizer
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
            is_with_sentiment=False
        ) 
        logging.info(f"--- Fold {i + 1} training completed ---")
        logging.info(f"Evaluating the performance of fold {i + 1} ...")
        f1_score, precision_score, recall_score, accuracy_score, auroc_score  = evaluate(test_dataloader, model, with_sentiment=False)
        logging.info(f"Evaluation of the performance of fold {i + 1} complete")
        f1_scores_without_sentiment.append(f1_score)
        precision_scores_without_sentiment.append(precision_score)
        recall_scores_without_sentiment.append(recall_score)
        accuracy_scores_without_sentiment.append(accuracy_score)
        auroc_scores_without_sentiment.append(auroc_score)
    logging.info("------ Kflod cross validation without sentiment completed ------")
    print()
    
    # Perform t-test
    logging.info("Performing ttest to check statistical significance")
    ttest_obj = stats.ttest_rel(f1_scores_with_sentiment, f1_scores_without_sentiment)
    logging.info(f'F1 t-test {ttest_obj}')
    ttest_manual = manual_paired_ttest(f1_scores_with_sentiment, f1_scores_without_sentiment, cv=10)
    logging.info(f'F1 t-test manual, stats: {ttest_manual[0]}, pvalue: {ttest_manual[1]}')
    
    ttest_obj = stats.ttest_rel(precision_scores_with_sentiment, precision_scores_without_sentiment)
    logging.info(f'Precision t-test {ttest_obj}')
    ttest_manual = manual_paired_ttest(precision_scores_with_sentiment, precision_scores_without_sentiment, cv=10)
    logging.info(f'Precision t-test manual, stats: {ttest_manual[0]}, pvalue: {ttest_manual[1]}')
    
    ttest_obj = stats.ttest_rel(recall_scores_with_sentiment, recall_scores_without_sentiment)
    logging.info(f'Recall t-test {ttest_obj}')
    ttest_manual = manual_paired_ttest(recall_scores_with_sentiment, recall_scores_without_sentiment, cv=10)
    logging.info(f'Recall t-test manual, stats: {ttest_manual[0]}, pvalue: {ttest_manual[1]}')
    
    ttest_obj = stats.ttest_rel(accuracy_scores_with_sentiment, accuracy_scores_without_sentiment)
    logging.info(f'Accuracy t-test {ttest_obj}')
    ttest_manual = manual_paired_ttest(accuracy_scores_with_sentiment, accuracy_scores_without_sentiment, cv=10)
    logging.info(f'Accuracy t-test manual, stats: {ttest_manual[0]}, pvalue: {ttest_manual[1]}')
    
    ttest_obj = stats.ttest_rel(auroc_scores_with_sentiment, auroc_scores_without_sentiment)
    logging.info(f'AUROC t-test {ttest_obj}')
    ttest_manual = manual_paired_ttest(auroc_scores_with_sentiment, auroc_scores_without_sentiment, cv=10)
    logging.info(f'AUROC t-test manual, stats: {ttest_manual[0]}, pvalue: {ttest_manual[1]}')
    
    logging.info(f'Mean F1 with sentiment: {np.mean(f1_scores_with_sentiment)}')
    logging.info(f'Mean precision with sentiment: {np.mean(precision_scores_with_sentiment)}')
    logging.info(f'Mean recall with sentiment: {np.mean(recall_scores_with_sentiment)}')
    logging.info(f'Mean auroc with sentiment: {np.mean(auroc_scores_with_sentiment)}')
    logging.info(f'Mean accuracy with sentiment: {np.mean(accuracy_scores_with_sentiment)}')
     
    logging.info(f'Mean F1 without sentiment: {np.mean(f1_scores_without_sentiment)}')
    logging.info(f'Mean precision without sentiment: {np.mean(precision_scores_without_sentiment)}')
    logging.info(f'Mean recall without sentiment: {np.mean(recall_scores_without_sentiment)}')
    logging.info(f'Mean auroc without sentiment: {np.mean(auroc_scores_without_sentiment)}')
    logging.info(f'Mean accuracy without sentiment: {np.mean(accuracy_scores_without_sentiment)}')
    logging.info("Ttest completed")
    

if __name__ == "__main__":
    log_dir ='./log_folder'
    get_logger(log_dir, get_filename() + CUSTOM_NAME)
    main()
