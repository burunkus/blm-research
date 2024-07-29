import os
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

seed_val = 42
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
                 max_length=100
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

        self.labels = []
        self.tweet_ids = []
        
        directory, file_name = os.path.split(data_path)
        file_extension = file_name.split('.')[-1]

        if file_extension == "txt":
            with open(data_path, encoding="utf-8") as file_handler:
                tweets = []
                for i, line in enumerate(file_handler):
                    tweet_id, tweet, label = line.split("\t")
                    tweet_id = tweet_id.strip()
                    tweet = tweet.strip()
                    label = label.strip()
                    tweets.append(tweet)
                    self.labels.append(int(label))
                    self.tweet_ids.append(tweet_id)
                #print(self.labels, self.tweet_ids)

            tokenized_tweet = tokenizer(tweets,
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
    Note: Code adapted from https://osf.io/qkjuv/
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
    Note: Code adapted from https://osf.io/qkjuv/
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
          classifier_criterion):

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
    Returns:
        None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    model.to(device)
    print("Model loaded!")

    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=1e-8)
    num_training_steps = len(train_dataloader) * epochs
    learning_rate_scheduler = get_scheduler("linear",
                                            optimizer=optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=num_training_steps)

    loss_values = []
    
    # For each epoch...
    for epoch in range(epochs):

        # Store true lables for global eval
        gold_labels = []

        # Store predicted labels for global eval
        predicted_labels = []

        # Perform one full pass over the training set.
        print(f'Epoch {epoch + 1} / {epochs}')
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

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

            # Perform a forward pass 
            outputs, _ = model(batch)
            
            loss = classifier_criterion(outputs.logits, batch_labels)

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
            
            # Progress update every 1000 batches.
            if step % 1000 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print(f'Batch {step} of {len(train_dataloader)}. Loss {total_loss / 1000}. Elapsed: {elapsed}.')
                

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {format_time(time.time() - t0)}")

        # Validation

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
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
                # Forward pass
                outputs, _ = model(batch)
                logits = outputs.logits 
                
                val_loss = classifier_criterion(outputs.logits, batch_labels)
                running_val_loss += val_loss
                
            predictions = nn.functional.softmax(logits, dim=-1)
            # Move probabilities to CPU
            predictions = predictions.detach().cpu().numpy()
            predictions_flattened = np.argmax(predictions, axis=1).flatten()
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            num_eval_steps += 1

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()

            # Store gold labels single list
            gold_labels.extend(labels_flat)
            # Store predicted labels single list
            predicted_labels.extend(pred_flat)
        
        avg_val_loss = running_val_loss / len(test_dataloader)
        print('LOSS train {} valid {}'.format(avg_train_loss, avg_val_loss))
        
        # Report the final accuracy for this validation run.
        print(f"  Validation Accuracy: {(eval_accuracy / num_eval_steps):.2f}")
        print(f"  Validation took: {format_time(time.time() - t0)}")
            
        print("")
        print("Evaluation on full prediction per epoch!")
        print(f"{classification_report(gold_labels, predicted_labels, digits=4)}")

    name = save_path.split('/')[-2] + '.pth'
    torch.save(model.state_dict(), save_path + name)
    print(f"model {name} saved at {save_path}")
    
    
def main():
    pass

if __name__ == "__main__":
    main()