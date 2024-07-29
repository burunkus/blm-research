import torch

# Download the dataset here: https://competitions.codalab.org/competitions/17751#learn_the_details-datasets
DATA_PATH = "../../../../../../zfs/socbd/eokpala/blm_research/data/"
SAVE_PATH = "../../../../../../zfs/socbd/eokpala/blm_research/models/emotion/"
CUSTOM_NAME = ""
NUMBER_OF_LABELS = 11

def parameters_for_fine_tuning():
    return {
        'criterion': torch.nn.BCEWithLogitsLoss()
    }


def parameter_for_sentiment_model():
    return {
        'num_labels': 2,
        'use_as_feature_extractor': True
    }


def parameter_for_emotion_model(): 
    return {
        'activation': [torch.nn.ReLU(), torch.nn.LeakyReLU(0.1), torch.nn.Tanh()],
        'input_size': 768,
        'hidden1': 768,
        'hidden2': 768,
        'hidden3': 768,
        'hidden4': 768,
        'dropout_rate': 0.1,
        'criterion': torch.nn.BCEWithLogitsLoss()
    }


def hyperparameters_for_early_stopping():
    #batch_size = 32
    batch_size = 8
    learning_rate = 1e-5
    epochs = 100
    patience = 5
    #patience = 7
    return batch_size, learning_rate, epochs, patience


def hyperparameters():
    batch_size = 8
    #batch_size = 16
    learning_rate = 1e-5
    epochs = 5
    return batch_size, learning_rate, epochs

