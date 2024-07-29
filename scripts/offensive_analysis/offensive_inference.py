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
    hyperparameters,
)
sys.path.append('/home/eokpala/blm-research/scripts')
from fine_tuning_module import CustomBertModel, OffensiveNetworkWithOneFullyConnectedLayer

seed_val = 23
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
            tokenizer (Object): BERT/RoBERTa tokenization object
            data_path (String): Absolute path to the train/test dataset
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
        self.emotions = []
        
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
                emotion = [
                    int(float(line[9].strip())),
                    int(float(line[10].strip())),
                    int(float(line[11].strip())),
                    int(float(line[12].strip())),
                    int(float(line[13].strip())),
                    int(float(line[14].strip())),
                    int(float(line[15].strip())),
                    int(float(line[16].strip())),
                    int(float(line[17].strip())),
                    int(float(line[18].strip())),
                    int(float(line[19].strip()))
                ]
                self.emotions.append(emotion)

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
                        self.genders[index],
                        torch.tensor(self.emotions[index])
                    )
        return dataset_item
    

def classify(dataloader, tokenizer, model, num_labels, save_as):
    '''
    Classify the sentiment of tweets in dataloader
    Args:
        dataloader (Iterable): A PyTorch iterable object through a dataset
        tokenizer (Object): BERT based tokenizer
        model (Object): The BERT based model
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
    
    absolute_path, file_name = os.path.split(save_as)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
    
    # Save the classification result
    with open(save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for step, batch in enumerate(dataloader):
            # Unpack the inputs from our dataloader
            batch_input_ids, batch_input_masks, batch_token_type_ids, batch_tweet_ids, batch_author_ids, batch_tweets, batch_dates, batch_names, batch_usernames, batch_sentiments, batch_races, batch_genders, batch_emotions = batch
            # Add batch to GPU/CPU except for tweet_ids since it is not a tensor
            batch_input_ids = batch_input_ids.to(device)
            batch_input_masks = batch_input_masks.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)

            batch = (batch_input_ids, batch_input_masks, None)

            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                outputs, logits = model(batch) # Call forward() in offensive network

            # Get probabilities
            outputs = outputs.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            
            predictions = np.argmax(outputs, axis=1).flatten()
            # Save the classification result
            for i, prediction in enumerate(predictions):
                anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, suprise, trust = [emotion.item() for emotion in batch_emotions[i]] #unpack tensor([emotion_i, ..., emotion_n])
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
                    trust,
                    prediction
                ]
                csv_handler.writerow(row)

            if step % 1000 == 0:
                logging.info(f"Processed batch {step} of {len(dataloader)}")
                
    
def main(hate_model_path, sentiment_model_path, trained_offensive_model_path, num_labels, batch_size, data, save_as):
    '''classify tweets into offensive or non-offensive classes'''
    
    logging.info("Construction the pretrained offensive model graph ...")
    hate_model = CustomBertModel(hate_model_path, num_labels)
    logging.info("Pretrained offensive model graph constructed.")
    
    logging.info("Constructing the sentiment model graph ...")
    sentiment_model = CustomBertModel(sentiment_model_path, num_labels)
    logging.info("Sentiment model graph constructed.")
    
    logging.info("Constructing OffensiveNetwork graph ...")
    model = OffensiveNetworkWithOneFullyConnectedLayer(hate_model, sentiment_model, num_labels, use_mean_repr=True)
    logging.info("OffensiveNetwork graph constructed ...")
    tokenizer = hate_model.tokenizer

    logging.info("Loading the fine-tuned offensive network weights ...")
    model.load_state_dict(torch.load(trained_offensive_model_path))
    logging.info("Fine-tuned offensive network weights loaded ...")
    
    logging.info("Loading dataset ...")
    data_instance = CustomTextDataset(tokenizer, data)
    dataloader = DataLoader(data_instance, batch_size=batch_size, shuffle=True)
    logging.info("Dataset loaded.")
    
    logging.info("Inference started ...")
    classify(dataloader, tokenizer, model, num_labels, save_as)
    logging.info("Inference completed.")


if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename() + CUSTOM_NAME)
    
    batch_size = 16
    num_labels = NUMBER_OF_LABELS
    
    hate_model_path = "vinai/bertweet-base"
    sentiment_model_path = "vinai/bertweet-base"
    trained_offensive_model_path = SAVE_PATH + "fine_tuned_bertweet_with_one_fc_layer_mean_repr_with_batch_size16_epochs5_max_len128/fine_tuned_fine_tuned_bertweet_with_one_fc_layer_mean_repr_with_batch_size16_epochs5_max_len128/fine_tuned_fine_tuned_bertweet_with_one_fc_layer_mean_repr_with_batch_size16_epochs5_max_len128.pth"
    data = DATA_PATH + "unique_tweets_2021_preprocessed_with_sentiment_race_gender_and_emotion.csv"
    save_as = DATA_PATH + "unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label.csv"
    main(hate_model_path, sentiment_model_path, trained_offensive_model_path, num_labels, batch_size, data, save_as)