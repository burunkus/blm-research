import json
import os
import random
import csv
import re
import sys
import numpy as np
import pandas as pd
import logging
import logging.handlers
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib import pyplot
from collections import defaultdict


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


def offensive_and_non_offensive_comments(file_path, dataset='twitter'):
    """
    Return tuple of list of the number of offensive comments and non-offensive comments
    """
    num_offensive, num_non_offensive = 0, 0
    offensive_map = defaultdict(int)
    non_offensive_map = defaultdict(int)
    
    with open(file_path) as csv_file_path_handle:
        csv_reader = csv.reader(csv_file_path_handle, delimiter=',')
        for i, row in enumerate(csv_reader):
            if dataset == 'parler':
                date = row[2]
                anger = row[4]
                disgust = row[6]
                fear = row[7]
                joy = row[8]
                optimism = row[10]
                label = row[15]
            elif dataset == 'twitter':
                date = row[3]
                anger = row[9]
                disgust = row[11]
                fear = row[12]
                joy = row[13]
                optimism = row[15]
                label = row[20]

            if label == '1':
                offensive_map[date] += 1
            elif label == '0':
                non_offensive_map[date] += 1

    return offensive_map, non_offensive_map
                    

def emotions_of_offensive_and_non_offensive_comments(file_path, dataset='twitter'):
    """
    Return tuple of list of the number of offensive comments and non-offensive comments
    """
    anger_offensive_map, anger_non_offensive_map = defaultdict(int), defaultdict(int)
    disgust_offensive_map, disgust_non_offensive_map = defaultdict(int), defaultdict(int)
    fear_offensive_map, fear_non_offensive_map = defaultdict(int), defaultdict(int)
    joy_offensive_map, joy_non_offensive_map = defaultdict(int), defaultdict(int)
    optimism_offensive_map, optimism_non_offensive_map = defaultdict(int), defaultdict(int)
    
    with open(file_path) as csv_file_path_handle:
        csv_reader = csv.reader(csv_file_path_handle, delimiter=',')
        for j, row in enumerate(csv_reader):
            if dataset == 'parler':
                date = row[2]
                anger = row[4]
                disgust = row[6]
                fear = row[7]
                joy = row[8]
                optimism = row[10]
                label = row[15]
            elif dataset == 'twitter':
                date = row[3]
                anger = row[9]
                disgust = row[11]
                fear = row[12]
                joy = row[13]
                optimism = row[15]
                label = row[20]
                
            if label == '1':
                # Anger emotion
                if anger == '1':
                    anger_offensive_map[date] += 1
                # Disgust emotion
                if disgust == '1':
                    disgust_offensive_map[date] += 1
                # Fear emotion
                if fear == '1':
                    fear_offensive_map[date] += 1
                # Joy emotion       
                if joy == '1':
                    joy_offensive_map[date] += 1
                # Optimism emotion
                if optimism == '1':
                    optimism_offensive_map[date] += 1
            elif label == '0':
                # Anger emotion
                if anger == '1':
                    anger_non_offensive_map[date] += 1
                # Disgust emotion
                if disgust == '1':
                    disgust_non_offensive_map[date] += 1
                # Fear emotion
                if fear == '1':
                    fear_non_offensive_map[date] += 1
                # Joy emotion       
                if joy == '1':
                    joy_non_offensive_map[date] += 1
                # Optimism emotion
                if optimism == '1':
                    optimism_non_offensive_map[date] += 1
                
    return (
        anger_offensive_map,
        disgust_offensive_map,
        fear_offensive_map,
        joy_offensive_map,
        optimism_offensive_map,
    ), (
        anger_non_offensive_map,
        disgust_non_offensive_map,
        fear_non_offensive_map,
        joy_non_offensive_map,
        optimism_non_offensive_map
    )


def ks_significance_test(twitter_list, parler_list):
    test_obj = stats.kstest(twitter_list, parler_list)
    return test_obj.statistic, test_obj.pvalue


def plot_offensive_and_non_offensive(twitter_data_map, parler_data_map):
    
    num_offensive_parler = sum(list(parler_data_map['offensive'].values()))
    num_non_offensive_parler = sum(list(parler_data_map['non-offensive'].values()))
    num_offensive_twitter = sum(list(twitter_data_map['offensive'].values()))
    num_non_offensive_twitter = sum(list(twitter_data_map['non-offensive'].values()))
    
    # Offensive plot 
    offensive_twitter = sorted(twitter_data_map['offensive'].items())
    offensive_dates_twitter = [np.datetime64(date) for date, _ in offensive_twitter]
    offensive_numbers_twitter = [num for _, num in offensive_twitter]
    
    offensive_parler = sorted(parler_data_map['offensive'].items())
    offensive_dates_parler = [np.datetime64(date) for date, _ in offensive_parler]
    offensive_number_parler = [num for _, num in offensive_parler]
    
    fig, ax = plt.subplots()
    month_year_formatter = mdates.DateFormatter('%b %Y')
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right') 
        
    # timeseries plot
    line1, = ax.plot(offensive_dates_twitter, offensive_numbers_twitter, "-b", label="Twitter", linewidth=1)
    line2, = ax.plot(offensive_dates_parler, offensive_number_parler, "-r", label="Parler", linewidth=1)
    
    # Add horizontal grid lines
    #plt.yaxis.grid(True)
    #ax.set_xlabel('Topic years')
    ax.legend()
    ax.set_ylabel('# Tweets')
    
    fig.savefig('figures/offensive_twitter_and_parler.png', format='png', bbox_inches='tight')
    
    # close plot
    plt.close()
    
    # Non-offensive plot 
    non_offensive_twitter = sorted(twitter_data_map['non-offensive'].items())
    non_offensive_dates_twitter = [np.datetime64(date) for date, _ in non_offensive_twitter]
    non_offensive_numbers_twitter = [num for _, num in non_offensive_twitter]
    
    non_offensive_parler = sorted(parler_data_map['non-offensive'].items())
    non_offensive_dates_parler = [np.datetime64(date) for date, _ in non_offensive_parler]
    non_offensive_number_parler = [num for _, num in non_offensive_parler]
    
    fig, ax = plt.subplots()
    month_year_formatter = mdates.DateFormatter('%b %Y')
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
        
    # timeseries plot
    line1, = ax.plot(non_offensive_dates_twitter, non_offensive_numbers_twitter, "-b", label="Twitter", linewidth=1)               
    line2, = ax.plot(non_offensive_dates_parler, non_offensive_number_parler, "-r", label="Parler", linewidth=1)
    
    # Add horizontal grid lines
    #ax.yaxis.grid(True)
    #ax.set_xlabel('Topic years')
    ax.legend()
    ax.set_ylabel('# Tweets')
    
    fig.savefig('figures/non_offensive_twitter_and_parler.png', format='png', bbox_inches='tight')
    
    # close plot
    plt.close()
    

def plot_emotions_of_offensive_and_non_offensive_comments(twitter_data_map, parler_data_map, emotion, offensive=True):
    
    twitter = sorted(twitter_data_map.items())
    dates_twitter = [np.datetime64(date) for date, _ in twitter]
    numbers_twitter = [num for _, num in twitter]
    
    parler = sorted(parler_data_map.items())
    dates_parler = [np.datetime64(date) for date, _ in parler]
    number_parler = [num for _, num in parler]
    
    fig, ax = plt.subplots()
    month_year_formatter = mdates.DateFormatter('%b %Y')
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right') 
        
    # timeseries plot
    line1, = ax.plot(dates_twitter, numbers_twitter, "-b", label="Twitter", linewidth=1)
    line2, = ax.plot(dates_parler, number_parler, "-r", label="Parler", linewidth=1)

    # Add horizontal grid lines
    #ax.yaxis.grid(True)
    #ax.set_xlabel('Topic years')
    ax.legend()
    ax.set_ylabel('# Tweets')
    
    if offensive:
        fig.savefig(f'figures/{emotion}_offensive_twitter_and_parler.png', format='png', bbox_inches='tight')
    else:
        fig.savefig(f'figures/{emotion}_non_offensive_twitter_and_parler.png', format='png', bbox_inches='tight')
    
    # close plot
    plt.close()
    
    
def main():

    twitter_dataset = "../../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_compliant_offensive_and_compliant_non_offensive.csv"
    parler_dataset = "../../../../../../zfs/socbd/eokpala/blm_research/data/parler_blm_preprocessed_2020_emotion_label.csv"

    # Offensive and Non-offensive of Twitter and Parler
    twitter_offensive_map, twitter_non_offensive_map = offensive_and_non_offensive_comments(twitter_dataset, dataset='twitter')
    parler_offensive_map, parler_non_offensive_map = offensive_and_non_offensive_comments(parler_dataset, dataset='parler')
    
    # KS-test
    twitter_offensive_list = [val for _, val in sorted(list(twitter_offensive_map.items()))]
    parler_offensive_list = [val for _, val in sorted(list(parler_offensive_map.items()))]
    statistic, pvalue = ks_significance_test(twitter_offensive_list, parler_offensive_list)
    logging.info(f'KS significant test for Twitter and Parler offensive comments, statistic: {statistic}, pvalue: {pvalue}')
    print()
    
    twitter_non_offensive_list = [val for _, val in sorted(list(twitter_non_offensive_map.items()))]
    parler_non_offensive_list = [val for _, val in sorted(list(parler_non_offensive_map.items()))]
    statistic, pvalue = ks_significance_test(twitter_non_offensive_list, parler_non_offensive_list)
    logging.info(f'KS significant test for Twitter and Parler non-offensive comments, statistic: {statistic}, pvalue: {pvalue}')
    print()
    
    # Plot number of offensive and non-offensive comments of Twitter and Parler
    twitter_data_map = {'offensive': twitter_offensive_map, 'non-offensive': twitter_non_offensive_map}
    parler_data_map = {'offensive': parler_offensive_map, 'non-offensive': parler_non_offensive_map}
    plot_offensive_and_non_offensive(twitter_data_map, parler_data_map)

    # Offensive and non-offensive comments emotions of Twitter and Parler 
    twitter_offensive_emotions, twitter_non_offensive_emotions = emotions_of_offensive_and_non_offensive_comments(twitter_dataset, dataset='twitter')
    parler_offensive_emotions, parler_non_offensive_emotions = emotions_of_offensive_and_non_offensive_comments(parler_dataset, dataset='parler')
    
    emotions = ['anger', 'disgust', 'fear', 'joy', 'optimism']
    for i, (twitter_emotion_map, parler_emotion_map) in enumerate(zip(twitter_offensive_emotions, parler_offensive_emotions)):
        emotion = emotions[i]
        # KS-test
        twitter_emotion_offensive_list = [val for _, val in sorted(list(twitter_emotion_map.items()))]
        parler_emotion_offensive_list = [val for _, val in sorted(list(parler_emotion_map.items()))]
        statistic, pvalue = ks_significance_test(twitter_emotion_offensive_list, parler_emotion_offensive_list)
        logging.info(f'KS significant test for Twitter and Parler {emotion.upper()} emotion in offensive comments, statistic: {statistic}, pvalue: {pvalue}')
        print()
        
        # Plot emotion
        plot_emotions_of_offensive_and_non_offensive_comments(twitter_emotion_map, parler_emotion_map, emotion, offensive=True)
    
    for i, (twitter_emotion_map, parler_emotion_map) in enumerate(zip(twitter_non_offensive_emotions, parler_non_offensive_emotions)):
        emotion = emotions[i]
        twitter_emotion_non_offensive_list = [val for _, val in sorted(list(twitter_emotion_map.items()))]
        parler_emotion_non_offensive_list = [val for _, val in sorted(list(parler_emotion_map.items()))]
        statistic, pvalue = ks_significance_test(twitter_emotion_non_offensive_list, parler_emotion_non_offensive_list)
        logging.info(f'KS significant test for Twitter and Parler {emotion.upper()} emotion non-offensive comments, statistic: {statistic}, pvalue: {pvalue}')
        print()
        
        # Plot emotion
        plot_emotions_of_offensive_and_non_offensive_comments(twitter_emotion_map, parler_emotion_map, emotion, offensive=False)
    
    
if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
    main()