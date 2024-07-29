import json
import sys
sys.path.append('/home/eokpala/blm-research/scripts')
from utils import *


def main():
    #file_location = "../../../../../zfs/socbd/eokpala/blm_research/data/tweets_2022.jsonl"
    #count_tweets(file_location)
    
    # Combine 2020, 2021, and 2022 datasets
    #parent_folder = "../../../../../zfs/socbd/eokpala/blm_research/data/"
    #dataset_names = ["tweets_2020.jsonl", "tweets_2021.jsonl", "tweets_2022.jsonl"]
    #save_as = "combined_datasets.jsonl"
    #combine_yearly_datasets(parent_folder, dataset_names, save_as)
    
    #file_name = "../../../../../zfs/socbd/eokpala/blm_research/data/combined_datasets.jsonl"
    #view_tweets(file_name)
    
    # Extract tweetID, authorID, and tweet text from combined dataset
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/combined_datasets.jsonl" 
    #save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/extracted_tweets_for_sampling.txt"
    #extract_tweets(file_path, save_as)
    
    # Sample 100,000 tweets to be passed through Perspective API to determine potential offensive tweets to label
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/extracted_tweets_for_sampling.txt"
    #save_sampled_as = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets.txt"
    #sample_tweets(file_path, save_sampled_as, k=100000)
    
    # Handle and preprocess Sentiment140 dataset
    #file_path = '../../../../../zfs/socbd/eokpala/blm_research/data/training.1600000.processed.noemoticon.csv'
    #save_as = '../../../../../zfs/socbd/eokpala/blm_research/data/sentiment140_preprocessed.txt'
    #handle_sentiment_dataset(file_path, save_as)
    #count_tweets(save_as)
    
    # Divide the sentiment140 dataset into train and test set
    #file_path = '../../../../../zfs/socbd/eokpala/blm_research/data/sentiment140_preprocessed.txt'
    #save_train_as = '../../../../../zfs/socbd/eokpala/blm_research/data/sentiment140_train.txt'
    #save_test_as = '../../../../../zfs/socbd/eokpala/blm_research/data/sentiment140_test.txt'
    #split_dataset(file_path, save_train_as, save_test_as, train_size=0.90)
    #count_tweets(save_train_as)
    #count_tweets(save_test_as)
    # Remove the duplicates that entered as a result of rerunning perspective withput deleting the previous file
    #file_location = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_with_toxicity_score_original.txt"
    #count_tweets(file_location)
    #save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_with_toxicity_score.txt"
    #remove_duplicate_tweets(file_location, save_as)
    #count_tweets(save_as)
    
    # Extract potential offensive tweets using just Perspective API
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_with_toxicity_score.txt"
    #save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_with_toxicity_score_greater_or_equal_to_0.7.txt"
    #extract_potential_offensive_tweets(file_path, save_as, threshold=0.7)
    #count_tweets(save_as)
    
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_with_toxicity_score.txt"
    #save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_with_toxicity_score_greater_or_equal_to_0.8.txt"
    #extract_potential_offensive_tweets(file_path, save_as, threshold=0.8)
    #count_tweets(save_as)
    
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_with_toxicity_score.txt"
    #save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_with_toxicity_score_greater_or_equal_to_0.9.txt"
    #extract_potential_offensive_tweets(file_path, save_as, threshold=0.9)
    #count_tweets(save_as)
    
    # Extract tweets with negative(0) sentiment and count them
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_having_toxicity_score_greater_or_equal_to_0.7_with_sentiment_labels.txt"
    #save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_having_toxicity_score_greater_or_equal_to_0.7_with_negative_sentiment.txt"
    #extract_tweets_with_negative_sentiment(file_path, save_as)
    #count_tweets(save_as)
    
    # To make labeling convenient, put the toxic tweets predicted to be of negative sentiment in a csv file
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_having_toxicity_score_greater_or_equal_to_0.7_with_negative_sentiment.txt"
    #save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_having_toxicity_score_greater_or_equal_to_0.7_with_negative_sentiment.csv"
    #write_tweets_with_negative_sentiment_to_csv(file_path, save_as)
     
    # Extract unique tweets in combined dataset and save tweets according to the year they were tweeted 
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/combined_datasets.jsonl"
    #save_combined_as = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_combined_datasets.csv"
    #save_preprocessed_combined_as = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_combined_datasets_preprocessed.csv"
    #save_2020_as = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020.csv"
    #save_preprocessed_2020_as = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed.csv"
    #save_2021_as = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2021.csv"
    #save_preprocessed_2021_as = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2021_preprocessed.csv"
    #save_2022_as = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2022.csv"
    #save_preprocessed_2022_as = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2022_preprocessed.csv"
    #retrieve_unique_tweets_from_combined_dataset(file_path, 
    #                                             save_combined_as,
    #                                             save_preprocessed_combined_as,
    #                                             save_2020_as,
    #                                             save_preprocessed_2020_as,
    #                                             save_2021_as,
    #                                             save_preprocessed_2021_as,
    #                                             save_2022_as,
    #                                             save_preprocessed_2022_as)
    
    # Count tweets after preprocessing - url, user, > 3 words, punctuations, stopwords, hashtag, elipses, number, duplicates
    #count_tweets_in_csv(save_preprocessed_2020_as)
    #count_tweets_in_csv(save_preprocessed_2021_as)
    #count_tweets_in_csv(save_preprocessed_2022_as)
    
    # Preprocess emotion dataset
    #train_file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/E-c/2018-E-c-En-train.txt"
    #save_preprocessed_train_as = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_preprocessed_train.csv"
    #handle_emotion_dataset(train_file_path, save_preprocessed_train_as)
    #test_file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/E-c/2018-E-c-En-test-gold.txt"
    #save_preprocessed_test_as = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_preprocessed_test.csv"
    #handle_emotion_dataset(test_file_path, save_preprocessed_test_as)
    #dev_file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/E-c/2018-E-c-En-dev.txt"
    #save_preprocessed_dev_as = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_preprocessed_dev.csv"
    #handle_emotion_dataset(dev_file_path, save_preprocessed_dev_as)
    
    # Infer sentiments of each year using the sentiment model in /sentiment_analysis/ folder
    
    # The 2020 file with sentiment is too big to run to completion within the 3 days limit of the high computing cluster, so break it.
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment.csv"
    #count_tweets(file_path)
    #path = "../../../../../zfs/socbd/eokpala/blm_research/data/"
    #save_as = "unique_tweets_2020_preprocessed_with_sentiment"
    #num_tweets = 4000000 # write 4M tweets in each file since total number of tweets is 16.5M to make it reasonable for race inference
    #divide_datasets_for_faster_race_inference(path, file_path, num_tweets, save_as)
    
    # Count the number of tweets. Splits should equal the number of tweets in unique_tweets_2020_preprocessed_with_sentiment.csv
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment.csv"
    #count_tweets(file_path)
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_split_0.csv"
    #count_tweets(file_path)
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_split_1.csv"
    #count_tweets(file_path)
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_split_2.csv"
    #count_tweets(file_path)
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_split_3.csv"
    #count_tweets(file_path)
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_split_4.csv"
    #count_tweets(file_path)
    
    #file_location = "../../../../../zfs/socbd/eokpala/blm_research/data/combined_datasets.jsonl"
    #find_invalid_names(file_location)
    
    # Annotated BLM dataset
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/offensive_dataset.csv"
    #save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/offensive_dataset_preprocessed.csv"
    #handle_offensive_blm_dataset(file_path, save_as)
    #count_offensive_tweets(save_as)
    
    # Split the offensive dataset into train and test dataset
    #data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/offensive_dataset_preprocessed.csv"
    #save_train_as = "../../../../../zfs/socbd/eokpala/blm_research/data/offensive_dataset_preprocessed_train.csv"
    #save_test_as = "../../../../../zfs/socbd/eokpala/blm_research/data/offensive_dataset_preprocessed_test.csv"
    #split_offensive_dataset(data_path, save_train_as, save_test_as)
    #count_offensive_tweets(save_train_as)
    #count_offensive_tweets(save_test_as)
    
    # Split the BLM dataset using 80:20 ratio
    # Split the offensive dataset into train and test dataset
    # data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/offensive_dataset_preprocessed.csv"
    # save_train_as = "../../../../../zfs/socbd/eokpala/blm_research/data/offensive_dataset_preprocessed_train_80_20_split.csv"
    # save_test_as = "../../../../../zfs/socbd/eokpala/blm_research/data/offensive_dataset_preprocessed_test_80_20_split.csv"
    # split_offensive_dataset(data_path, save_train_as, save_test_as, train_size=0.80)
    # count_offensive_tweets(save_train_as)
    # count_offensive_tweets(save_test_as)

    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2021_preprocessed.csv"
    #count_tweets(file_path)
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2022_preprocessed.csv"
    #count_tweets(file_path)
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_preprocessed_train.csv"
    #count_tweets(file_path)
    #file_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_preprocessed_test.csv"
    #count_tweets(file_path)
    #file_path ='../../../../../zfs/socbd/eokpala/blm_research/data/sentiment140_preprocessed.txt'
    #count_tweets(file_path)
    #file_path = '../../../../../zfs/socbd/eokpala/blm_research/data/sentiment140_train.txt'
    #count_tweets(file_path)
    #file_path = '../../../../../zfs/socbd/eokpala/blm_research/data/sentiment140_test.txt'
    #count_tweets(file_path)
    
    # Infer race using RaceBERT
    # Infer gender using NeuralGenderDemographer
    # Infer emotions using the emotion model in /emotion_analysis/ folder
    # Infer offensive and non-offensive using the offensive model in /offensive_analysis/ folder
    
    '''
    data_paths = [
        "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_split_0.csv",
        "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_split_1.csv",
        "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_split_2.csv",
        "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_split_3.csv",
        "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_split_4.csv"
    ]
    save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined.csv"
    combine_split_2020_datasets(data_paths, save_as)
    '''
    
    # Count Offensive and Non-offensive tweets
    #file_path_2020 = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined.csv"
    #count_offensive_non_offensive_tweets(file_path_2020)
    #file_path_2021 = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label.csv"
    #count_offensive_non_offensive_tweets(file_path_2021)
    #file_path_2022 = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label.csv"
    #count_offensive_non_offensive_tweets(file_path_2022)
    
    # Split dataset by offensive, non-offensive, race, gender, and intersectionality
    #data_paths = ["../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined.csv", "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label.csv", "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label.csv"]
    #extract_tweets_based_on_demographics(data_paths)
    
    # Sample 2M tweets. Some 2020 non-offensive tweets datasets have > 2M tweets
    # The number of non-offensive tweets in 2020 is large so reduce it to reduce memory requirement of BERTopic
    #non_offensive = "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_non_offensive.csv"
    #datasets = [non_offensive]
    #sample_from_non_offensive_tweets(datasets)
    
    # Sample 2M tweets from the 2021 non-offensive tweet which contains 3M tweets.
    #non_offensive = "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_non_offensive.csv"
    #datasets = [non_offensive]
    #sample_from_non_offensive_tweets(datasets, k=2000000)
    
    # Handle compliant tweets. Extract tweet ids of non-compliant tweets using compliant tweets.py
    
    # Extract compliant tweets from 2020 offensive tweets
    #non_compliant_ids_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/non_compliant_2020_offensive_tweets.txt"
    #offensive_tweets_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_offensive.csv"
    #save_as = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_offensive_compliant.csv"
    #extract_compliant_tweets(non_compliant_ids_file_path, offensive_tweets_file_path, save_as)
    
    # Extract compliant tweets from 2021 offensive tweets
    #non_compliant_ids_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/non_compliant_2021_offensive_tweets.txt"
    #offensive_tweets_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_offensive.csv"
    #save_as = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_offensive_compliant.csv"
    #extract_compliant_tweets(non_compliant_ids_file_path, offensive_tweets_file_path, save_as)
    
    # Extract compliant tweets from 2022 offensive tweets
    #non_compliant_ids_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/non_compliant_2022_offensive_tweets.txt"
    #offensive_tweets_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label_offensive.csv"
    #save_as = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label_offensive_compliant.csv"
    #extract_compliant_tweets(non_compliant_ids_file_path, offensive_tweets_file_path, save_as)
    
    # Extract compliant tweets from 2020 non-offensive tweets
    #non_compliant_ids_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/non_compliant_2020_non_offensive_tweets.txt"
    #non_offensive_tweets_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_non_offensive.csv"
    #save_as = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_non_offensive_compliant.csv"
    #extract_compliant_tweets(non_compliant_ids_file_path, non_offensive_tweets_file_path, save_as)
    
    # Extract compliant tweets from 2021 non-offensive tweets
    #non_compliant_ids_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/non_compliant_2021_offensive_tweets.txt"
    #non_offensive_tweets_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_non_offensive.csv"
    #save_as = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_non_offensive_compliant.csv"
    #extract_compliant_tweets(non_compliant_ids_file_path, non_offensive_tweets_file_path, save_as)
    
    # Extract compliant tweets from 2022 non-offensive tweets
    #non_compliant_ids_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/non_compliant_2022_offensive_tweets.txt"
    #non_offensive_tweets_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label_non_offensive.csv"
    #save_as = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label_non_offensive_compliant.csv"
    #extract_compliant_tweets(non_compliant_ids_file_path, non_offensive_tweets_file_path, save_as)
    
    # Extract compliant tweets from sampled 2020 non-offensive tweets
    #non_compliant_ids_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/non_compliant_2020_non_offensive_tweets.txt"
    #non_offensive_tweets_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_non_offensive_sampled.csv"
    #save_as = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_non_offensive_sampled_compliant.csv"
    #extract_compliant_tweets(non_compliant_ids_file_path, non_offensive_tweets_file_path, save_as)
    
    # Extract compliant tweets from sampled 2021 non-offensive tweets
    #non_compliant_ids_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/non_compliant_2021_offensive_tweets.txt"
    #non_offensive_tweets_file_path = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_non_offensive_sampled.csv"
    #save_as = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_non_offensive_sampled_compliant.csv"
    #extract_compliant_tweets(non_compliant_ids_file_path, non_offensive_tweets_file_path, save_as)
    
    # Combine compliant offensive and non-offensive tweets
    #     compliant_non_offensive_tweets_2020 = "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_non_offensive_compliant.csv" # each line is of the form tweet_id_author_id,tweet
#     compliant_offensive_tweets_2020 = "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_offensive_compliant.csv" # each line is of the form tweet_id_author_id,tweet
#     all_2020_tweets = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined.csv"
#     save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_compliant_offensive_and_compliant_non_offensive.csv"
#     combine_compliant_tweets(compliant_offensive_tweets_2020, compliant_non_offensive_tweets_2020, all_2020_tweets, save_as)
#     count_offensive_non_offensive_tweets(save_as)
    
#     compliant_non_offensive_tweets_2021 = "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_non_offensive_compliant.csv"
#     compliant_offensive_tweets_2021 = "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_offensive_compliant.csv"
#     all_2021_tweets = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label.csv"
#     save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_combined_compliant_offensive_and_compliant_non_offensive.csv"
#     combine_compliant_tweets(compliant_offensive_tweets_2021, compliant_non_offensive_tweets_2021, all_2021_tweets, save_as)
#     count_offensive_non_offensive_tweets(save_as)
  
    #compliant_offensive_file = "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label_offensive_compliant.csv"
    #compliant_non_offensive_file = "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label_non_offensive_compliant.csv"
    #all_2022_tweets = "../../../../../zfs/socbd/eokpala/blm_research/data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label.csv"
    #save_as = "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label_combined_compliant.csv"
    #combine_compliant_tweets(compliant_offensive_file, compliant_non_offensive_file, all_2022_tweets, save_as)
    #count_offensive_non_offensive_tweets(save_as)

    # Plot emotions and daily offensive and non-offensive tweets using run_analysis.py
    
    # Perform network analysis of users in /user_analysis/ folder
    
    # Plot emotions of offenders and recipients tweets using run_analysis.py
    
    # Perform topic modeling using BERTopic in /topic_modeling/ folder
    
    # Handle Parler comments
    # file_path = "../../../../../../zfs/socbd/eokpala/blm_research/data/parler_blm.csv"
    # save_as = "../../../../../../zfs/socbd/eokpala/blm_research/data/parler_blm_preprocessed.csv"
    # view_parler_tweets(file_path, number_of_texts=10)
    # handle_parler_dataset(file_path, save_as)
    # count_tweets_in_csv(save_as)
    
    # Extract only 2020 comments from Parler
    # file_path = "../../../../../../zfs/socbd/eokpala/blm_research/data/parler_blm_preprocessed.csv"
    # save_as = "../../../../../../zfs/socbd/eokpala/blm_research/data/parler_blm_preprocessed_2020.csv"
    # extract_parler_comments_by_year(file_path, save_as, year='2020')
    # count_tweets_in_csv(save_as)
    
    # Perform cross platform analysis in /cross_platform/ folder
    
    # Perform topic modeling and evaluation in /topic_modeling/ folder
    
    
if __name__ == "__main__":
    main()
