import os
import sys
import torch
import time
import csv
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
sys.path.append('/home/eokpala/blm-research/scripts')
from utils import preprocess_tweet


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
        
        self.ids = []
        self.tweets = []
        self.labels = []
        directory, file_name = os.path.split(data_path)
        file_extension = file_name.split('.')[-1]

        with open(data_path) as csv_file_handle:
            csv_reader = csv.reader(csv_file_handle, delimiter=',')
            for i, line in enumerate(csv_reader):
                self.ids.append(line[0].strip())
                self.tweets.append(line[1].strip())
                emotion = [int(e) for e in line[2:]]
                self.labels.append(torch.tensor(emotion, dtype=torch.float32)) #make labels float32 

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
                        self.labels[index]
                        #torch.tensor(self.labels[index], dtype=torch.float32) #make labels float32 
                    )
        return dataset_item
    

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Credit: https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        
def flat_accuracy(preds, labels):
    """
    Calculate the accuracy using the predicted values and the true labels
    Args:
        preds: ndarray of model predictions
        labels: ndarray of true labels
    Returns:
        accuracy (ndarray): accuracy of the current batch
    """
    
    accuracy = accuracy_score(labels, preds)
    return accuracy


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def plot_loss_vs_epoch(train_loss, valid_loss):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(train_loss) + 1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig('loss_plot.png', bbox_inches='tight')


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
          is_bare_model=False,
          is_layers_frozen=False,
          patience=None):

    """
    Fine-tune a new model using the pre-trained model and save the new model in
    save_path

    Args:
        train_dataloader (Object): A PyTorch iterable object through the train set
        test_dataloader (Object): A PyTorch iterable object through the test set
        model_path (String): The location (absolute) of the pre-trained model
        num_labels (Int): The number of classes in our task
        learning_rate (Float): Learning rate for the optimizer
        epochs(Int): The number of times to go through the entire dataset
        save_path (String): Absolute path where the fine-tuned model will be saved
        is_layers_frozen (Boolean): Whether the sentiment part of the network is frozen. Defaults to True
        is_bare_model (Boolean): Whether the simple BERT based model is being used. Defaults to False
        patience (Int): The parameter used to control early stopping. How many consecutive decline in performance needs to be observed before stopping training. Defaults to None i.e., no early stopping is performed. 
    Returns:
        None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    
    if is_layers_frozen == True:
        optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()),
                          lr=learning_rate,
                          eps=1e-8)
    else:
        optimizer = AdamW(model.parameters(),
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
    
    # intialize early stopping object
    if patience is not None:
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path+name, trace_func=logging.info)
    
    # Store the average loss after each epoch so we can plot the learning curve i.e loss vs epoch.
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
            if is_bare_model:
                outputs, _ = model(batch)
                logits = outputs.logits
                outputs = torch.sigmoid(logits).round()
            else:
                # This will return outputs after sigmoid and logits from the output layer before sigmoid
                outputs, logits = model(batch)
                
            # Calculate the loss
            loss = criterion(logits, batch_labels)
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end.
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
                # Forward pass, calculate logit predictions.
                if is_bare_model:
                    outputs, _ = model(batch)
                    logits = outputs.logits
                    outputs = torch.sigmoid(logits).round()
                else:
                    # This will return outputs after sigmoid and logits before sigmoid from the output layer
                    outputs, logits = model(batch)
                
                # Calculate the loss
                val_loss = criterion(logits, batch_labels)
                running_val_loss += val_loss.item()
            
            # Move predictions, logits and labels to CPU
            predictions = outputs.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(predictions, label_ids)
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            num_eval_steps += 1

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
        
        target_names = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "suprise", "trust"]
        logging.info(f"\n{classification_report(gold_labels, predicted_labels, target_names=target_names, digits=4)}")
        
        # Check if the validation loss is no longer improving(i.e decreasing) after "patience" consecutive times
        if patience is not None:
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

    torch.save(model.state_dict(), save_path + name)
    logging.info(f"model {name} saved at {save_path}")

    
def main():
    pass


if __name__ == "__main__":
    main()