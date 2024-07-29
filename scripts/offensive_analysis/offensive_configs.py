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


def hyperparameters():
    batch_size = 16
    learning_rate = 1e-5
    epochs = 5
    return batch_size, learning_rate, epochs

