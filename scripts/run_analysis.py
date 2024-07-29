import csv
import os
from analysis import *
from utils import extract_tweets_based_on_demographics, count_extracted_tweets_based_on_demographics, sample_from_non_offensive_tweets, extract_compliant_tweets, combine_compliant_and_non_compliant_tweets, count_offensive_non_offensive_tweets

def main():
    
    # Check correlation between anger and disgust in non-offensive tweets for all years
    #data_path = "emotions_of_non_offensive_tweets.csv"
    #test_correlation_between_anger_and_disgust(data_path)
    
    # Check correlation between anger and disgust in offensive tweets for all years
    #data_path = "emotions_of_offensive_tweets.csv"
    #test_correlation_between_anger_and_disgust(data_path)
    
    # Check correlation between anger and disgust in non-offensive tweets for 2020
    #data_path = "emotions_of_non_offensive_tweets_2020.csv"
    #test_correlation_between_anger_and_disgust(data_path)
    
    # Check correlation between anger and disgust in offensive tweets for 2020
    #data_path = "emotions_of_offensive_tweets_2020.csv"
    #test_correlation_between_anger_and_disgust(data_path)
    
    # Plot emotions in tweets and daily tweets
    # data_paths =["/zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_compliant_offensive_and_compliant_non_offensive.csv", "/zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_combined_compliant_offensive_and_compliant_non_offensive.csv", "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label_combined_compliant.csv"]
    # plot_emotions2(data_paths)

    # Plot emotions in tweets by offenders and receivers 
    # data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/offenders_offensive_and_non_offensive_tweets.csv"
    # plot_emotions_offenders_receivers([data_path], save_as='_offenders')
    # data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/receivers_offensive_and_non_offensive_tweets.csv"
    # plot_emotions_offenders_receivers([data_path], save_as='_receivers')
    
    # Vertical lines. The line positions are obtained by first running change point detection algorithm on the data using change_point_detection.py. Second, looking at raw data to see where the most change occurred. Third, add the dates with the most change points in the arrays
    
#     events_lines = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']
#     emotions_in_offensive_tweets_vlines_offenders = [np.datetime64('2020-05-31'), 
#                                                      np.datetime64('2020-08-24'), 
#                                                      np.datetime64('2021-01-02'), 
#                                                      np.datetime64('2021-04-17'), 
#                                                      np.datetime64('2021-11-23'), 
#                                                      np.datetime64('2022-05-23')]
#     emotions_in_non_offensive_tweets_vlines_offenders = [np.datetime64('2020-05-31'), 
#                                                          np.datetime64('2020-08-24'), 
#                                                          np.datetime64('2020-09-18'), 
#                                                          np.datetime64('2021-01-02'), 
#                                                          np.datetime64('2021-04-17'), 
#                                                          np.datetime64('2021-11-23'), 
#                                                          np.datetime64('2022-02-22'), 
#                                                          np.datetime64('2022-05-23')]
    
#     data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/offenders_offensive_and_non_offensive_tweets_no_overlap.csv"
#     save_as = '_offenders_no_overlap'
#     plot_emotions_offenders_receivers([data_path], 
#                                       save_as,
#                                       events_lines,
#                                       emotions_in_offensive_tweets_vlines_offenders,
#                                       emotions_in_non_offensive_tweets_vlines_offenders)
    
#     emotions_in_offensive_tweets_vlines_receivers = [np.datetime64('2020-05-31'), 
#                                                      np.datetime64('2020-08-24'), 
#                                                      np.datetime64('2021-01-02'), 
#                                                      np.datetime64('2021-04-22'),
#                                                      np.datetime64('2021-11-23')
#                                                      ]
#     emotions_in_non_offensive_tweets_vlines_receivers = [np.datetime64('2020-05-31'), 
#                                                          np.datetime64('2020-08-24'), 
#                                                          np.datetime64('2020-09-18'), 
#                                                          np.datetime64('2021-01-02'), 
#                                                          np.datetime64('2021-04-17'), 
#                                                          np.datetime64('2021-11-23'), 
#                                                          np.datetime64('2022-02-22'), 
#                                                          np.datetime64('2022-05-23')]
#     data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/receivers_offensive_and_non_offensive_tweets_no_overlap.csv"
#     save_as = '_receivers_no_overlap'
#     plot_emotions_offenders_receivers([data_path], 
#                                       save_as,
#                                       events_lines,
#                                       emotions_in_offensive_tweets_vlines_receivers,
#                                       emotions_in_non_offensive_tweets_vlines_receivers)
    
    # Plot emotions in tweets by high offenders and high receivers with a threshold of 50
#     data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/offenders_offensive_and_non_offensive_tweets_high_50.csv"
#     plot_emotions_offenders_receivers([data_path], save_as='_high_offenders_threshold_50')
#     data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/receivers_offensive_and_non_offensive_tweets_high_50.csv"
#     plot_emotions_offenders_receivers([data_path], save_as='_high_receivers_threshold_50')
    
#     data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/offenders_offensive_and_non_offensive_tweets_high_50_no_overlap.csv"
#     plot_emotions_offenders_receivers([data_path], save_as='_high_offenders_threshold_50_no_overlap')
#     data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/receivers_offensive_and_non_offensive_tweets_high_50_no_overlap.csv"
#     plot_emotions_offenders_receivers([data_path], save_as='_high_receivers_threshold_50_no_overlap')
        
    # Plot emotions in tweets by high offenders and high receivers with a threshold of 100
#     data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/offenders_offensive_and_non_offensive_tweets_high_100.csv"
#     plot_emotions_offenders_receivers([data_path], save_as='_high_offenders_threshold_100')
#     data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/receivers_offensive_and_non_offensive_tweets_high_100.csv"
#     plot_emotions_offenders_receivers([data_path], save_as='_high_receivers_threshold_100')
    
#     data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/offenders_offensive_and_non_offensive_tweets_high_100_no_overlap.csv"
#     plot_emotions_offenders_receivers([data_path], save_as='_high_offenders_threshold_100_no_overlap')
#     data_path = "../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/receivers_offensive_and_non_offensive_tweets_high_100_no_overlap.csv"
#     plot_emotions_offenders_receivers([data_path], save_as='_high_receivers_threshold_100_no_overlap')

    # Offensive and non-offensive comments in Parler
    # file_path = "../../../../../../zfs/socbd/eokpala/blm_research/data/parler_blm_preprocessed_2020_emotion_label.csv"
    # count_offensive_non_offensive_tweets(file_path)
    
    

if __name__ == "__main__":
    main()