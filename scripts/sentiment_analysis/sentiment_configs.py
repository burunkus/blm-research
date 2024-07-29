import torch

# Download the dataset from here: http://help.sentiment140.com/for-students
DATA_PATH = "../../../../../zfs/socbd/eokpala/blm_research/data/"
SAVE_PATH = "../../../../../zfs/socbd/eokpala/blm_research/models/sentiment/"
CUSTOM_NAME = ""
USE_CUSTOM_BERT = True
NUMBER_OF_LABELS = 2

def custom_bert_parameters():
    classifier_criterion = torch.nn.CrossEntropyLoss()
    return classifier_criterion

def custom_bert_with_fc_parameters():
    return {
        'classifier_criterion': torch.nn.CrossEntropyLoss(),
        'dropout_rate': 0.1,
        }
      
def hyperparameters():
    batch_size = 256
    learning_rate = 1e-5
    epochs = 5
    return batch_size, learning_rate, epochs

