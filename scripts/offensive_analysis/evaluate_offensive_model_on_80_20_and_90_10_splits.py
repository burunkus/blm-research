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
from sklearn.utils import resample
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
    parameters_of_offensive_model_with_fc_layers,
    parameters_of_offensive_model_with_deep_attention
)
from fine_tune_offensive_utils import flat_accuracy, format_time, train
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
            #print(line)
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
    
    logging.info(classification_report(gold_labels, predicted_labels, digits=4))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
    logging.info(f"\nMacro F1: {macro_f1:.3f}")
    logging.info(f"Macro Precision: {macro_precision:.3f}")
    logging.info(f"Macro Recall: {macro_recall:.3f}")
    logging.info(f"Micro F1: {micro_f1:.3f}")
    logging.info(f"Micro Precision: {micro_precision:.3f}")
    logging.info(f"Micro Recall: {micro_recall:.3f}")
    
    # AUROC score
    greater_label_probabilities = []
    for prob in probabilities:
        greater_label_probabilities.append(prob[1])
    roc_output_value = roc_auc_score(gold_labels, greater_label_probabilities)
    logging.info(f'AUROC score: {roc_output_value}')
    
    return macro_f1, roc_output_value
    
    
def main(hate_model_path, 
         sentiment_model_path, 
         trained_offensive_model_path, 
         batch_size, 
         test_dataset_90_10, 
         test_dataset_80_20, 
         num_labels):
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
    
    f1_scores_90_10_split = []
    auroc_scores_90_10_split = []
    f1_scores_80_20_split = []
    auroc_scores_80_20_split = []
    
    original_dataset_90_10_split = []
    with open(test_dataset_90_10, encoding="utf-8-sig") as csv_file_handle:
        csv_reader = csv.reader(csv_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader):
            row = []
            row.append(line[0].strip())
            row.append(line[1].strip())
            row.append(line[2].strip())
            row.append(int(line[3].strip()))
            original_dataset_90_10_split.append(row)
            
    original_dataset_80_20_split = []
    with open(test_dataset_80_20, encoding="utf-8-sig") as csv_file_handle:
        csv_reader = csv.reader(csv_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader):
            row = []
            row.append(line[0].strip())
            row.append(line[1].strip())
            row.append(line[2].strip())
            row.append(int(line[3].strip()))
            original_dataset_80_20_split.append(row)
    
    num_iterations = 1000
    sample_size = len(original_dataset_90_10_split)
    
    # Run boostrap for the 90:10 split
    for i in range(num_iterations):
        # Get a boostrap sample 
        data = resample(original_dataset_90_10_split, replace=True, n_samples=sample_size)
        
        logging.info(f"Loading dataset of boostrap iteration {i} ...")
        data_instance = CustomTextDataset(tokenizer, data)
        dataloader = DataLoader(data_instance, batch_size=batch_size, shuffle=True)
        logging.info("Boostrap dataset loaded.")
    
        logging.info(f"Evaluation started for boostrap iteration {i} ...")
        f1, auroc = evaluate(dataloader, model)
        f1_scores_90_10_split.append(f1)
        auroc_scores_90_10_split.append(auroc)
        logging.info("Evaluation completed.")
        
    # Run boostrap for the 80:20 split
    for i in range(num_iterations):
        # Get a boostrap sample 
        data = resample(original_dataset_80_20_split, replace=True, n_samples=sample_size)
        
        logging.info(f"Loading dataset of boostrap iteration {i} ...")
        data_instance = CustomTextDataset(tokenizer, data)
        dataloader = DataLoader(data_instance, batch_size=batch_size, shuffle=True)
        logging.info("Boostrap dataset loaded.")
    
        logging.info(f"Evaluation started for boostrap iteration {i} ...")
        f1, auroc = evaluate(dataloader, model)
        f1_scores_80_20_split.append(f1)
        auroc_scores_80_20_split.append(auroc)
        logging.info("Evaluation completed.")
        
    # Save the histogram plot of the performance metrics
    plt.hist(f1_scores_90_10_split)
    plt.savefig('histograms/f1_90_10_split.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    plt.hist(auroc_scores_90_10_split)
    plt.savefig('histograms/auroc_90_10_split.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    plt.hist(f1_scores_80_20_split)
    plt.savefig('histograms/f1_80_20_split.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    plt.hist(auroc_scores_80_20_split)
    plt.savefig('histograms/auroc_80_20_split.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    # Difference calculation
    f1_scores_difference = np.asarray(f1_scores_90_10_split) - np.asarray(f1_scores_80_20_split)
    auroc_scores_difference = np.asarray(auroc_scores_90_10_split) - np.asarray(auroc_scores_80_20_split)
    
    plt.hist(f1_scores_difference)
    plt.savefig('histograms/f1_difference_split.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    plt.hist(auroc_scores_difference)
    plt.savefig('histograms/auroc_difference_split.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    # Confidence intervals 
    alpha = 0.95
    
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(f1_scores_difference, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(f1_scores_difference, p))
    logging.info(f'{alpha * 100} confidence interval for the difference in F1 scores of the 90:10 and 80:20 split is {lower * 100} and {upper * 100}')
    
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(auroc_scores_difference, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(auroc_scores_difference, p))
    logging.info(f'{alpha * 100} confidence interval for the difference in AUROC scores of the 90:10 and 80:20 split is {lower * 100} and {upper * 100}')
    
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(f1_scores_90_10_split, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(f1_scores_90_10_split, p))
    logging.info(f'{alpha * 100} confidence interval for F1 scores of 90:10 split is {lower * 100} and {upper * 100}')
    
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(f1_scores_80_20_split, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(f1_scores_80_20_split, p))
    logging.info(f'{alpha * 100} confidence interval for F1 scores of 80:20 split is {lower * 100} and {upper * 100}')
    
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(auroc_scores_90_10_split, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(auroc_scores_90_10_split, p))
    logging.info(f'{alpha * 100} confidence interval for AUROC scores of 90:10 split is {lower * 100} and {upper * 100}')
    
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(auroc_scores_80_20_split, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(auroc_scores_80_20_split, p))
    logging.info(f'{alpha * 100} confidence interval for AUROC scores of 80:20 split is {lower * 100} and {upper * 100}')
    

if __name__ == "__main__":
    log_dir ='./log_folder'
    get_logger(log_dir, get_filename())
    
    batch_size = 16
    num_labels = NUMBER_OF_LABELS
    
    hate_model_path = "vinai/bertweet-base"
    sentiment_model_path = "vinai/bertweet-base"
    trained_offensive_model_path = SAVE_PATH + "fine_tuned_bertweet_with_one_fc_layer_mean_repr_with_batch_size16_epochs5_max_len128/fine_tuned_fine_tuned_bertweet_with_one_fc_layer_mean_repr_with_batch_size16_epochs5_max_len128/fine_tuned_fine_tuned_bertweet_with_one_fc_layer_mean_repr_with_batch_size16_epochs5_max_len128.pth"
    test_dataset_90_10 = DATA_PATH + "offensive_dataset_preprocessed_test.csv"
    test_dataset_80_20 = DATA_PATH + "offensive_dataset_preprocessed_test_80_20_split.csv"
    main(hate_model_path, sentiment_model_path, trained_offensive_model_path, batch_size, test_dataset_90_10, test_dataset_80_20, num_labels)
