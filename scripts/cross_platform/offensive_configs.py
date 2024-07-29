import torch

DATA_PATH = "../../../../../../zfs/socbd/eokpala/blm_research/data/"
SAVE_PATH = "../../../../../../../zfs/socbd/eokpala/blm_research/models/offensive/"
CUSTOM_NAME = ""
NUMBER_OF_LABELS = 2
USE_MEAN_REPR = False

def parameter_for_sentiment_model():
    return {
        'num_labels': 2,
        'use_as_feature_extractor': True
    }


def parameters_for_offensive_model():
    return {
        'criterion': torch.nn.CrossEntropyLoss()
    }


def parameters_of_offensive_model_with_fc_layers():
    return {
        'activation': [torch.nn.ReLU(), torch.nn.LeakyReLU(0.1), torch.nn.Tanh()],
        'hidden1': 768,
        'hidden2': 768,
        'hidden3': 768,
        'hidden4': 2, 
        'criterion': torch.nn.CrossEntropyLoss()
    }#'hidden4': 2 is for code convenience, it is not used
    
    
def parameters_of_offensive_model_with_deep_attention():
    return {
        'activation': [torch.nn.ReLU(), torch.nn.LeakyReLU(0.1), torch.nn.Tanh()],
        'hidden1': 768,
        'hidden2': 768,
        'hidden3': 768,
        'hidden4': 1,
        'criterion': torch.nn.CrossEntropyLoss()
    }


def hyperparameters_for_early_stopping():
    #batch_size = 32
    #batch_size = 16
    batch_size = 8
    learning_rate = 1e-5
    epochs = 100
    #patience = 5
    patience = 7
    return batch_size, learning_rate, epochs, patience


def hyperparameters():
    #batch_size = 32
    batch_size = 16
    #batch_size = 8
    learning_rate = 1e-5
    epochs = 5
    return batch_size, learning_rate, epochs

