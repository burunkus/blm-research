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
from sentiment_configs import (
    NUMBER_OF_LABELS,
    DATA_PATH,
    SAVE_PATH,
    USE_CUSTOM_BERT,
    custom_bert_with_fc_parameters,
    hyperparameters, 
    custom_bert_parameters
)
sys.path.append('/home/eokpala/blm-research/scripts')
from fine_tuning_module import CustomBertModel
from utils import preprocess_sentiment_tweet

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

class CustomTextDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data_path,
                 is_preprocessed=False,
                 padding="max_length",
                 truncation=True,
                 max_length=100
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
        self.raw_tweets = []
        self.tweets = []
        self.is_preprocessed = is_preprocessed

        directory, file_name = os.path.split(data_path)
        file_extension = file_name.split('.')[-1]

        if file_extension == "txt":
            with open(data_path, encoding="utf-8") as file_handler:
                for line in file_handler:
                    tweet_id, author_id, tweet = line.split("\t")
                    self.raw_tweets.append(tweet.strip())
                    self.tweet_ids.append(tweet_id.strip())
                    self.author_ids.append(author_id.strip())
                    # The sampled 100K tweets have not been preprocessed, proprocess them here
                    if self.is_preprocessed == False:
                        tweet = preprocess_sentiment_tweet(tweet)
                    self.tweets.append(tweet.strip())

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
                        self.raw_tweets[index],
                        self.tweets[index]
                    )
        return dataset_item
    

class CustomTextDatasetForTweets(Dataset):
    def __init__(self,
                 tokenizer,
                 data_path,
                 padding="max_length",
                 truncation=True,
                 max_length=100
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
                        self.usernames[index]
                    )
        return dataset_item
    
    
def classify(dataloader, tokenizer, model, num_labels, save_as):
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
    
    absolute_path, file_name = os.path.split(save_as)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
    
    # Save the classification result
    with open(save_as, 'a+') as file_handle:
        for batch in dataloader:
            # Unpack the inputs from our dataloader
            batch_input_ids, batch_input_masks, batch_token_type_ids, batch_tweet_ids, batch_author_ids, batch_raw_tweets, batch_tweets = batch
            # Add batch to GPU/CPU except for tweet_ids since it is not a tensor
            batch_input_ids = batch_input_ids.to(device)
            batch_input_masks = batch_input_masks.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)

            batch = (batch_input_ids, batch_input_masks, None)

            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                outputs, _ = model(batch)
                logits = outputs.logits  

            # Get probabilities
            predictions = nn.functional.softmax(logits, dim=-1)
            # Move probabilities to CPU
            predictions = predictions.detach().cpu().numpy()
            predictions_flattened = np.argmax(predictions, axis=1).flatten()

            # Save the classification result
            for i, prediction in enumerate(predictions_flattened):
                file_handle.write(batch_tweet_ids[i] + '\t' + batch_author_ids[i] + '\t' + batch_raw_tweets[i] + '\t' + str(prediction) + '\n')

        
def classify_all_tweets(dataloader, tokenizer, model, num_labels, save_as):
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
    
    absolute_path, file_name = os.path.split(save_as)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
    
    # Save the classification result
    with open(save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for batch in dataloader:
            # Unpack the inputs from our dataloader
            batch_input_ids, batch_input_masks, batch_token_type_ids, batch_tweet_ids, batch_author_ids, batch_tweets, batch_dates, batch_names, batch_usernames = batch
            # Add batch to GPU/CPU except for tweet_ids since it is not a tensor
            batch_input_ids = batch_input_ids.to(device)
            batch_input_masks = batch_input_masks.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)

            batch = (batch_input_ids, batch_input_masks, None)

            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                outputs, _ = model(batch)
                logits = outputs.logits 

            # Get probabilities
            predictions = nn.functional.softmax(logits, dim=-1)
            # Move probabilities to CPU
            predictions = predictions.detach().cpu().numpy()
            predictions_flattened = np.argmax(predictions, axis=1).flatten()

            # Save the classification result
            for i, prediction in enumerate(predictions_flattened):
                row = [batch_tweet_ids[i], batch_author_ids[i], batch_tweets[i], batch_dates[i], batch_names[i], batch_usernames[i], str(prediction)]
                csv_handler.writerow(row)


def classify_sampled_tweets(num_labels, batch_size, file_path):
    '''Predict the sentiment of 100K sampled tweets using the fine-tuned BERTweet model'''
    
    classifier_criterion = custom_bert_parameters()
    model = CustomBertModel(base_model_name,
                            num_labels)

    model.load_state_dict(torch.load(model_path))
    tokenizer = model.tokenizer
            
    data_path = file_path + "sampled_tweets_with_toxicity_score_greater_or_equal_to_0.7.txt"
    data = CustomTextDataset(tokenizer, data_path, is_preprocessed=False)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    save_as = file_path + "sampled_tweets_having_toxicity_score_greater_or_equal_to_0.7_with_sentiment_labels.txt"
    classify(dataloader, tokenizer, model, num_labels, save_as)
    

def classify_tweets(model_path, data_path, save_as, num_labels, batch_size):
    ''' Predict the sentiment of all tweets using the fine-tuned BERTweet model'''
    
    classifier_criterion = custom_bert_parameters()
    model = CustomBertModel(base_model_name, num_labels)
    model.load_state_dict(torch.load(model_path))
    tokenizer = model.tokenizer
            
    data = CustomTextDatasetForTweets(tokenizer, data_path)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    classify_all_tweets(dataloader, tokenizer, model, num_labels, save_as)
    
            
if __name__ == "__main__":
    batch_size = 32
    num_labels = NUMBER_OF_LABELS
    model_path = SAVE_PATH + "bertweet/fine_tuned_bertweet/fine_tuned_bertweet.pth"
    base_model_name = "vinai/bertweet-base"
    save_path = DATA_PATH
    
    # Classify the 100K tweets for 
    #classify_sampled_tweets(num_labels, batch_size, save_path)
    
    # Classify all tweets
    data_path = DATA_PATH + "unique_tweets_2020_preprocessed.csv"
    save_as = DATA_PATH + "unique_tweets_2020_preprocessed_with_sentiment.csv"
    classify_tweets(model_path, data_path, save_as, num_labels, batch_size)