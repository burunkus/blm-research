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
sys.path.append('/home/eokpala/blm-research/scripts')
from fine_tuning_module import CustomBertModel, BareBertModel, OffensiveNetworkWithDeepAttention2 as EmotionNetworkWithDeepAttention2

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

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
                 data_path,
                 padding="max_length",
                 truncation=True,
                 max_length=128
                 ):

        """
        Generate a single example and its label from data_path
        Args:
            tokenizer (Object): BERT variant tokenization object
            data_path (String): Absolute path to dataset. 
            Each line is of the form tweetID \t tweet \t label
            padding (String): How to padding sequences, defaults to "max_lenght"
            truncation (Boolean): Whether to truncate sequences, defaults to True
            max_length (Int): The maximum length of a sequence, sequence longer
            than max_length will be truncated and those shorter will be padded
        Retruns:
            dataset_item (Tuple): A tuple of tensors - tokenized text, attention mask and labels
        """

        if not os.path.exists(data_path):
            raise ValueError(f"Input file {data_path} does not exist")

        self.tweet_ids = []
        self.author_ids = []
        self.tweets = []
        self.dates = []
        self.names = []
        self.usernames = []
        self.sentiments = []
        self.races = []
        self.genders = []

        directory, file_name = os.path.split(data_path)
        file_extension = file_name.split('.')[-1]

        with open(data_path) as csv_file_handle:
            csv_reader = csv.reader(csv_file_handle, delimiter=',')
            for i, line in enumerate(csv_reader):
                self.tweet_ids.append(line[0].strip())
                self.author_ids.append(line[1].strip())
                self.tweets.append(line[2].strip())
                self.dates.append(line[3].strip())
                self.names.append(line[4].strip())
                self.usernames.append(line[5].strip())
                self.sentiments.append(line[6].strip())
                self.races.append(line[7].strip())
                self.genders.append(line[8].strip())

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
                        torch.tensor(self.token_type_ids[index]),
                        self.tweet_ids[index],
                        self.author_ids[index],
                        self.tweets[index],
                        self.dates[index],
                        self.names[index],
                        self.usernames[index],
                        self.sentiments[index],
                        self.races[index],
                        self.genders[index]
                    )
        return dataset_item
    

def classify(dataloader, tokenizer, model, save_as):
    '''
    Classify the sentiment of tweets in dataloader
    Args:
        dataloader (Iterable): A PyTorch iterable object through a dataset
        tokenizer (Object): BERT tokenizer
        model (Object): The BERT model
        save_as (String): Absolute path where the classified tweets will be saved
    Returns:
        None
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Put the model in evaluation mode
    model.eval()
    
    absolute_path, file_name = os.path.split(save_as)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
               
    # Save the classification result
    with open(save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for step, batch in enumerate(dataloader):
            # Unpack the inputs from our dataloader
            batch_input_ids, batch_input_masks, batch_token_type_ids, batch_tweet_ids,\
            batch_author_ids, batch_tweets, batch_dates, batch_names, batch_usernames,\
            batch_sentiments, batch_races, batch_genders = batch
            # Add batch to GPU/CPU except for tweet_ids since it is not a tensor
            batch_input_ids = batch_input_ids.to(device)
            batch_input_masks = batch_input_masks.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)

            batch = (batch_input_ids, batch_input_masks, None)

            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs, logits = model(batch) # Call forward() in the emotion model
            
            # Get probabilities
            outputs = torch.sigmoid(logits).round().detach().cpu().numpy()
            batch_size = batch_input_ids.size()[0]
            # Save the classification result having batch_size > 1 by looping through each example(datapoint) in 
            # the batch
            if batch_size > 1:
                for i, prediction in enumerate(outputs):
                    anger = prediction[0]
                    anticipation = prediction[1]
                    disgust = prediction[2]
                    fear = prediction[3]
                    joy = prediction[4]
                    love = prediction[5]
                    optimism = prediction[6]
                    pessimism = prediction[7]
                    sadness = prediction[8]
                    suprise = prediction[9]
                    trust = prediction[10]

                    row = [
                        batch_tweet_ids[i],
                        batch_author_ids[i],
                        batch_tweets[i],
                        batch_dates[i],
                        batch_names[i],
                        batch_usernames[i],
                        batch_sentiments[i],
                        batch_races[i],
                        batch_genders[i],
                        anger, 
                        anticipation, 
                        disgust, 
                        fear, 
                        joy, 
                        love, 
                        optimism, 
                        pessimism, 
                        sadness, 
                        suprise, 
                        trust
                    ]
                    csv_handler.writerow(row)
            # Handle the edge case where the last batch of dataloader has only 1 datapoint (or example) 
            # i.e batch_input_ids is of shape [1, max_length]
            elif batch_size == 1:
                anger = outputs[0]
                anticipation = outputs[1]
                disgust = outputs[2]
                fear = outputs[3]
                joy = outputs[4]
                love = outputs[5]
                optimism = outputs[6]
                pessimism = outputs[7]
                sadness = outputs[8]
                suprise = outputs[9]
                trust = outputs[10]
                
                row = [
                    batch_tweet_ids[0],
                    batch_author_ids[0],
                    batch_tweets[0],
                    batch_dates[0],
                    batch_names[0],
                    batch_usernames[0],
                    batch_sentiments[0],
                    batch_races[0],
                    batch_genders[0],
                    anger, 
                    anticipation, 
                    disgust, 
                    fear, 
                    joy, 
                    love, 
                    optimism, 
                    pessimism, 
                    sadness, 
                    suprise, 
                    trust
                ]
                csv_handler.writerow(row)
            
            if step % 1000 == 0:
                logging.info(f"Processed batch {step} of {len(dataloader)}")    
    

def main(base_model_path, trained_emotion_model_path, num_labels, batch_size, data, save_as):
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
    
    logging.info("Inference started ...")
    classify(dataloader, tokenizer, model, save_as)
    logging.info("Inference completed.")
    
            
if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename() + CUSTOM_NAME)
    
    batch_size = 8
    num_labels = NUMBER_OF_LABELS
    base_model_path = "vinai/bertweet-base"
    trained_emotion_model_path = SAVE_PATH + "fine_tuned_bertweet_on_emotion_and_sent_mean_repr_with_early_stopping_hidden4_256_patience7_with_max_len128_batch_size8_leaky_relu_ENWDA2/fine_tuned_bertweet_on_emotion_and_sent_mean_repr_with_early_stopping_hidden4_256_patience7_with_max_len128_batch_size8_leaky_relu_ENWDA2/fine_tuned_bertweet_on_emotion_and_sent_mean_repr_with_early_stopping_hidden4_256_patience7_with_max_len128_batch_size8_leaky_relu_ENWDA2.pth"
    data = DATA_PATH + "unique_tweets_2021_preprocessed_with_sentiment_race_and_gender.csv"
    save_as = DATA_PATH + "unique_tweets_2021_preprocessed_with_sentiment_race_gender_and_emotion.csv"
    main(base_model_path, trained_emotion_model_path, num_labels, batch_size, data, save_as)