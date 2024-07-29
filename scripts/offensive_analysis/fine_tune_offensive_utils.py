import os
import csv
import sys
import torch
import time
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import pickle
from _datetime import datetime as dt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch import nn
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

seed_val = 23
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


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
        self.labels = []
        directory, file_name = os.path.split(data_path)
        file_extension = file_name.split('.')[-1]

        with open(data_path) as csv_file_handle:
            csv_reader = csv.reader(csv_file_handle, delimiter=',')
            for i, line in enumerate(csv_reader):
                self.tweet_ids.append(line[0].strip())
                self.author_ids.append(line[1].strip())
                self.tweets.append(line[2].strip())
                self.labels.append(int(line[3].strip()))

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
        
        
def flat_accuracy(preds, labels):
    """
    Calculate the accuracy using the predicted values and the true labels
    Args:
        preds: ndarray of model predictions
        labels: ndarray of true labels
    Returns:
        accuracy (ndarray): accuracy of the current batch
    """

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def plot_auc_roc_curve(true_labels, predicted_labels, n_classes, save_as):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels, predicted_labels)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels, predicted_labels)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(f'auc_roc_{save_as}.png')
    
    
def train(train_dataloader,
          test_dataloader,
          tokenizer,
          model,
          num_labels,
          learning_rate,
          epochs,
          save_path,
          criterion,
          logging,
          is_with_sentiment=True):

    """
    Fine-tune a new model using the pre-trained model and save the new model in
    save_path

    Args:
        train_dataloader (Object): A PyTorch iterable object through the train set
        test_dataloader (Object): A PyTorch iterable object through the test set
        model (String): The location (absolute) of the pre-trained model
        num_labels (Int): The number of classes in our task
        learning_rate (Float): Learning rate for the optimizer
        epochs(Int): The number of times to go through the entire dataset
        save_path (String): Absolute path where the fine-tuned model will be saved
        criterion (Object): PyTorch optimization function
        logging (Logging): For logging purposes
        is_with_sentiment (Boolean): Indicates whether the offensive model is being fine-tuned using sentiment features. Defaults to true
    Returns:
        None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Optimize layers with requires_grad=True which is the hate model
    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()),
                      lr=learning_rate,
                      eps=1e-8)
    num_training_steps = len(train_dataloader) * epochs
    learning_rate_scheduler = get_scheduler("linear",
                                            optimizer=optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=num_training_steps)

    # path to save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    name = save_path.split('/')[-2] + '.pth'
    
    avg_train_losses = []
    avg_valid_losses = []
    
    # For each epoch...
    for epoch in range(1, epochs + 1):

        # Store true lables for global eval
        gold_labels = []
        # Store predicted labels for global eval
        predicted_labels = []

        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0.0
        total_train_accuracy = 0.0

        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Unpack this training batch from our dataloader.
            # `batch` contains three pytorch tensors:
            # [0]: input ids
            # [1]: attention masks
            # [2]: labels
            batch_input_ids = batch[0].to(device)
            batch_input_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            batch = (batch_input_ids, batch_input_mask, batch_labels)
            
            model.zero_grad()
            # Perform a forward pass (evaluate the model on this training batch).
            if is_with_sentiment:
                outputs, logits = model(batch)
            else: 
                outputs, _ = model(batch)
                logits = outputs.logits
                outputs = nn.functional.softmax(logits, dim=-1)
            
            # Calculate the loss
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()
            
            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            learning_rate_scheduler.step()
            
            # Get predictions and move predictions and labels to CPU
            predictions = outputs.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences.
            train_accuracy = flat_accuracy(predictions, label_ids)
            # Accumulate the total accuracy.
            total_train_accuracy += train_accuracy
            
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        # Store the loss value for plotting the learning curve.
        avg_train_losses.append(avg_train_loss)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        elapsed_train_time = format_time(time.time() - t0)

        # Validation 
        t0 = time.time()
        # Put the model in evaluation mode
        model.eval()
        # Tracking variables
        eval_accuracy = 0.0
        num_eval_steps, num_eval_examples = 0, 0
        running_val_loss = 0.0
        
        # Evaluate data for one epoch
        for batch in test_dataloader:
            # Add batch to GPU/CPU
            batch_input_ids = batch[0].to(device)
            batch_input_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            batch = (batch_input_ids, batch_input_mask, batch_labels)

            with torch.no_grad():
                if is_with_sentiment: 
                    outputs, logits = model(batch)
                else: 
                    outputs, _ = model(batch)
                    logits = outputs.logits
                    outputs = nn.functional.softmax(logits, dim=-1)
                
            # Calculate the loss
            val_loss = criterion(logits, batch_labels)
            # Accumulate validation loss
            running_val_loss += val_loss
            
            # Move probabilities to CPU
            predictions = outputs.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(predictions, label_ids)
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            num_eval_steps += 1
            
            predictions = np.argmax(predictions, axis=1).flatten()
            label_ids = label_ids.flatten()
            
            # Store gold labels single list
            gold_labels.extend(label_ids)
            # Store predicted labels single list
            predicted_labels.extend(predictions)
        
        elapsed_valid_time = format_time(time.time() - t0)
        avg_val_loss = running_val_loss / num_eval_steps
        avg_valid_losses.append(avg_val_loss)
        avg_valid_accuracy = eval_accuracy / num_eval_steps
        
        epoch_len = len(str(epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train loss: {avg_train_loss:.5f} ' +
                     f'valid loss: {avg_val_loss:.5f} ' +
                     f'train acc: {avg_train_accuracy:.5f} ' +
                     f'valid acc: {avg_valid_accuracy:.5f} ' +
                     f'train time: {elapsed_train_time} ' +
                     f'valid time: {elapsed_valid_time}')
        
        # Report the statistics for this epoch's validation run.
        logging.info(print_msg)
        
        target_names = ["non-offensive", "offensive"]
        logging.info(f"\n{classification_report(gold_labels, predicted_labels, target_names=target_names, digits=4)}")
        
    torch.save(model.state_dict(), save_path + name)
    logging.info(f"model {name} saved at {save_path}")
    
    
def main():
    pass

if __name__ == "__main__":
    main()