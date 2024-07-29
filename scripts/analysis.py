import json
import os
import random
import csv
import re
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib import pyplot

def count_offensive_and_non_offensive(file_paths):
    
    for file_path in file_paths:
        num_offensive_tweets = 0
        num_non_offensive_tweets = 0
        with open(file_path) as input_file_handle:
            csv_reader = csv.reader(input_file_handle, delimiter=',')
            for i, row in enumerate(csv_reader, 1):
                label = row[20]
                
                if label == '0':
                    num_non_offensive_tweets += 1
                elif label == '1':
                    num_offensive_tweets += 1
                
        print(f'File: {file_path}')
        print(f'Number of offensive tweets: {num_offensive_tweets}')
        print(f'Number of non offensive tweets: {num_non_offensive_tweets}')
    
    
def plot_offensive_and_non_offensive(datasets):
    '''
    Plot the number of tweets vs date of the offensive and non-offensive tweets 
    in out datasets.
    Args:
        datasets (List[String]): Contains the dataset path of each year
    Returns:
        None
    '''
    
    dates_offensive_map = {}
    dates_non_offensive_map = {}
    for file_path in datasets:
        with open(file_path) as csv_file_path_handle:
            csv_reader = csv.reader(csv_file_path_handle, delimiter=',')
            for i, row in enumerate(csv_reader):
                date = row[3]
                label = row[20]
                if label == '1':
                    if date in dates_offensive_map:
                        dates_offensive_map[date] += 1
                    else:
                        dates_offensive_map[date] = 1
                elif label == '0':
                    if date in dates_non_offensive_map:
                        dates_non_offensive_map[date] += 1
                    else:
                        dates_non_offensive_map[date] = 1
    
    offensive = sorted(dates_offensive_map.items())
    non_offensive = sorted(dates_non_offensive_map.items())
    
    offensive_dates, num_offensive_tweets = [np.datetime64(date) for date, _ in offensive], [num_tweets for _, num_tweets in offensive]
    non_offensive_dates, num_non_offensive_tweets = [np.datetime64(date) for date, _ in non_offensive], [num_tweets for _, num_tweets in non_offensive]
    
    # Plot
    month_year_formatter = mdates.DateFormatter('%b %Y')
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout='constrained')
    ax.xaxis.set_major_formatter(month_year_formatter)
    ax.plot(non_offensive_dates, num_non_offensive_tweets, label='non-offensive')
    ax.plot(offensive_dates, num_offensive_tweets, label='offensive')
    #ax.set_xlabel('x label')
    ax.set_ylabel('# of Tweets')
    ax.set_yscale('log')
    ax.set_title("Number of Daily Tweets")
    ax.legend()
    ax.grid(True)
    # Rotates and right aligns the x labels. 
    # Also moves the bottom of the axes up to make room for them.
    fig.autofmt_xdate()
    plt.savefig('figures/off_non_off_log_scale.png', bbox_inches='tight')
    

def test_correlation_between_anger_and_disgust(emotion_file_path):
    '''
    From the emotion distribution graph, anger and disgust seem to be correlated as 
    they follow the same pattern or have similar values. Test this correlation statistically.
    Arguments:
        emotion_file_path (String): The file containing the emotion values. This file is of the format
        Anger,
        date1, date2, ...
        value1, value2, ...
        Disgust, 
        ...
    Returns:
        None
    '''
    
    anger, disgust = [], []
    with open(emotion_file_path) as input_file_handle:
        csv_reader = csv.reader(input_file_handle, delimiter=',')
        for i, row in enumerate(csv_reader):
            # Get anger
            if i == 2:
                for value in row:
                    anger.append(float(value))
            elif i == 5:
                for value in row:
                    disgust.append(float(value))
                
    correlation_value, pvalue = stats.pearsonr(anger, disgust)
    print(f'Correlation between anger and disgust in {emotion_file_path}: \n{correlation_value}, p-value: {pvalue}\n')
    
    
def bot_distribution(file_path, year):
    
    # Number of accounts that have a specific bot score
    score_count_map = {}
    bot_scores = []
    with open(file_path) as csv_file_handle:
        csv_reader = csv.reader(csv_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader):
            tweet_id = line[0]
            user_id = line[1]
            user_name = line[2]
            bot_score = line[3]
            
            if bot_score != 'NA':
                bot_scores.append(float(bot_score))
                if bot_score in score_count_map:
                    score_count_map[bot_score] += 1
                else:
                    score_count_map[bot_score] = 1
    
    
    bot_scores = sorted(bot_scores)
    print(bot_scores)
    # Plot distribution of bot scores
    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout='constrained')
    bins = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    ax.hist(bot_scores, bins=bins, edgecolor='black', linewidth=0.5)
    #ax.grid(True)
    ax.set_ylabel('Number of Accounts')
    ax.set_xlabel('Bot Score')
    fig.tight_layout()
    plt.savefig(f'figures/{year}_bot_distribution.png', bbox_inches='tight', dpi=100)
    
    #with open(f'{year}_bot_distribution.csv', mode='w') as csv_file_handle:
    #    csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #    csv_handler.writerow(bot_scores)
    
       
def plot_emotions2(datasets):
    '''
    Plot the number of tweets vs date of the offensive and non-offensive tweets 
    in out datasets, proportion of offensive tweets by emotions, proportion of non-offensive tweets by emotions,
    bar chart of emotion distribution. 
    Args:
        datasets (List[String]): Contains the dataset path of each year
    Returns:
        None
    '''
    
    num_offensive, num_non_offensive = 0, 0
    num_offensive_2020, num_non_offensive_2020 = 0, 0
    num_offensive_2021, num_non_offensive_2021 = 0, 0
    num_offensive_2022, num_non_offensive_2022 = 0, 0
    # Daily tweets 
    dates_offensive_map = {}
    dates_non_offensive_map = {}
    # emotions 
    dates_anger_offensive_map, dates_anger_non_offensive_map = {}, {}
    dates_disgust_offensive_map, dates_disgust_non_offensive_map = {}, {}
    dates_fear_offensive_map, dates_fear_non_offensive_map = {}, {}
    dates_joy_offensive_map, dates_joy_non_offensive_map = {}, {}
    #dates_love_offensive_map, dates_love_non_offensive_map = {}, {}
    dates_optimism_offensive_map, dates_optimism_non_offensive_map = {}, {}
    #dates_sadness_offensive_map, dates_sadness_non_offensive_map = {}, {}
    
    for i, file_path in enumerate(datasets):
        with open(file_path) as csv_file_path_handle:
            csv_reader = csv.reader(csv_file_path_handle, delimiter=',')
            for j, row in enumerate(csv_reader):
                date = row[3]
                anger = row[9]
                disgust = row[11]
                fear = row[12]
                joy = row[13]
                #love = row[14]
                optimism = row[15]
                #sadness = row[17]
                label = row[20]
                if label == '1':
                    num_offensive += 1
                    if i == 0:
                        num_offensive_2020 += 1
                    elif i == 1:
                        num_offensive_2021 += 1
                    else:
                        num_offensive_2022 += 1
                        
                    # Daily tweets
                    if date in dates_offensive_map:
                        dates_offensive_map[date] += 1
                    else:
                        dates_offensive_map[date] = 1
                    # Anger emotion
                    if anger == '1':
                        if date in dates_anger_offensive_map:
                            dates_anger_offensive_map[date] += 1
                        else:
                            dates_anger_offensive_map[date] = 1
                    # Disgust emotion
                    if disgust == '1':
                        if date in dates_disgust_offensive_map:
                            dates_disgust_offensive_map[date] += 1
                        else:
                            dates_disgust_offensive_map[date] = 1
                    # Fear emotion
                    if fear == '1':
                        if date in dates_fear_offensive_map:
                            dates_fear_offensive_map[date] += 1
                        else:
                            dates_fear_offensive_map[date] = 1
                    # Joy emotion       
                    if joy == '1':
                        if date in dates_joy_offensive_map:
                            dates_joy_offensive_map[date] += 1
                        else:
                            dates_joy_offensive_map[date] = 1
                    # Optimism emotion
                    if optimism == '1':
                        if date in dates_optimism_offensive_map:
                            dates_optimism_offensive_map[date] += 1
                        else:
                            dates_optimism_offensive_map[date] = 1
                            
                elif label == '0':
                    num_non_offensive += 1
                    if i == 0:
                        num_non_offensive_2020 +=1
                    elif i == 1:
                        num_non_offensive_2021 += 1
                    else:
                        num_non_offensive_2022 += 1
                    
                    # Daily tweets
                    if date in dates_non_offensive_map:
                        dates_non_offensive_map[date] += 1
                    else:
                        dates_non_offensive_map[date] = 1
                    # Anger emotion
                    if anger == '1':
                        if date in dates_anger_non_offensive_map:
                            dates_anger_non_offensive_map[date] += 1
                        else:
                            dates_anger_non_offensive_map[date] = 1
                    # Disgust emotion
                    if disgust == '1':
                        if date in dates_disgust_non_offensive_map:
                            dates_disgust_non_offensive_map[date] += 1
                        else:
                            dates_disgust_non_offensive_map[date] = 1
                    # Fear emotion
                    if fear == '1':
                        if date in dates_fear_non_offensive_map:
                            dates_fear_non_offensive_map[date] += 1
                        else:
                            dates_fear_non_offensive_map[date] = 1
                    # Joy emotion       
                    if joy == '1':
                        if date in dates_joy_non_offensive_map:
                            dates_joy_non_offensive_map[date] += 1
                        else:
                            dates_joy_non_offensive_map[date] = 1
                    # Optimism emotion
                    if optimism == '1':
                        if date in dates_optimism_non_offensive_map:
                            dates_optimism_non_offensive_map[date] += 1
                        else:
                            dates_optimism_non_offensive_map[date] = 1
    
    print(f'Number of overall offensive: {num_offensive}, overall non-offensive: {num_non_offensive}')
    print(f'Number of offensive in 2020: {num_offensive_2020}, non-offensive in 2020: {num_non_offensive_2020}')
    print(f'Number of offensive in 2021: {num_offensive_2021}, non-offensive in 2021: {num_non_offensive_2021}')
    print(f'Number of offensive in 2022: {num_offensive_2022}, non-offensive in 2022: {num_non_offensive_2022}')
    print()
    
    # Daily tweets
    offensive = sorted(dates_offensive_map.items())
    non_offensive = sorted(dates_non_offensive_map.items())
    offensive_dates, num_offensive_tweets = [np.datetime64(date) for date, _ in offensive], [num_tweets for _, num_tweets in offensive]
    non_offensive_dates, num_non_offensive_tweets = [np.datetime64(date) for date, _ in non_offensive], [num_tweets for _, num_tweets in non_offensive]
    print(f'Offensive dates: {offensive_dates[:5]}')
    print(f'# offensive dates: {num_offensive_tweets[:5]}')
    print(f'Non-offensive dates: {non_offensive_dates[:5]}')
    print(f'# non-offensive dates: {num_non_offensive_tweets[:5]}')
    print()
    
    # Anger emotions
    anger_in_offensive = sorted(dates_anger_offensive_map.items())
    anger_in_non_offensive = sorted(dates_anger_non_offensive_map.items())
    anger_in_offensive_dates, num_anger_in_offensive_tweets = [], []
    anger_in_non_offensive_dates, num_anger_in_non_offensive_tweets = [], []
    anger_in_offensive_dates_2020, num_anger_in_offensive_tweets_2020 = [], []
    anger_in_non_offensive_dates_2020, num_anger_in_non_offensive_tweets_2020 = [], []
    anger_in_offensive_dates_2021, num_anger_in_offensive_tweets_2021 = [], []
    anger_in_non_offensive_dates_2021, num_anger_in_non_offensive_tweets_2021 = [], []
    anger_in_offensive_dates_2022, num_anger_in_offensive_tweets_2022 = [], []
    anger_in_non_offensive_dates_2022, num_anger_in_non_offensive_tweets_2022 = [], []
    
    for date, num_tweets in anger_in_offensive:
        anger_in_offensive_dates.append(np.datetime64(date))
        num_anger_in_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            anger_in_offensive_dates_2020.append(np.datetime64(date))
            num_anger_in_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            anger_in_offensive_dates_2021.append(np.datetime64(date))
            num_anger_in_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            anger_in_offensive_dates_2022.append(np.datetime64(date))
            num_anger_in_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
            
    for date, num_tweets in anger_in_non_offensive:
        anger_in_non_offensive_dates.append(np.datetime64(date))
        num_anger_in_non_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            anger_in_non_offensive_dates_2020.append(np.datetime64(date))
            num_anger_in_non_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            anger_in_non_offensive_dates_2021.append(np.datetime64(date))
            num_anger_in_non_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            anger_in_non_offensive_dates_2022.append(np.datetime64(date))
            num_anger_in_non_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
        
    # Disgust emotion
    disgust_in_offensive = sorted(dates_disgust_offensive_map.items())
    disgust_in_non_offensive = sorted(dates_disgust_non_offensive_map.items())
    disgust_in_offensive_dates, num_disgust_in_offensive_tweets = [], []
    disgust_in_non_offensive_dates, num_disgust_in_non_offensive_tweets = [], []
    disgust_in_offensive_dates_2020, num_disgust_in_offensive_tweets_2020 = [], []
    disgust_in_non_offensive_dates_2020, num_disgust_in_non_offensive_tweets_2020 = [], []
    disgust_in_offensive_dates_2021, num_disgust_in_offensive_tweets_2021 = [], []
    disgust_in_non_offensive_dates_2021, num_disgust_in_non_offensive_tweets_2021 = [], []
    disgust_in_offensive_dates_2022, num_disgust_in_offensive_tweets_2022 = [], []
    disgust_in_non_offensive_dates_2022, num_disgust_in_non_offensive_tweets_2022 = [], []
    
    for date, num_tweets in disgust_in_offensive:
        disgust_in_offensive_dates.append(np.datetime64(date))
        num_disgust_in_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            disgust_in_offensive_dates_2020.append(np.datetime64(date))
            num_disgust_in_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            disgust_in_offensive_dates_2021.append(np.datetime64(date))
            num_disgust_in_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            disgust_in_offensive_dates_2022.append(np.datetime64(date))
            num_disgust_in_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
           
    for date, num_tweets in disgust_in_non_offensive:
        disgust_in_non_offensive_dates.append(np.datetime64(date))
        num_disgust_in_non_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            disgust_in_non_offensive_dates_2020.append(np.datetime64(date))
            num_disgust_in_non_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            disgust_in_non_offensive_dates_2021.append(np.datetime64(date))
            num_disgust_in_non_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            disgust_in_non_offensive_dates_2022.append(np.datetime64(date))
            num_disgust_in_non_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
            
    # Fear emotion
    fear_in_offensive = sorted(dates_fear_offensive_map.items())
    fear_in_non_offensive = sorted(dates_fear_non_offensive_map.items())
    fear_in_offensive_dates, num_fear_in_offensive_tweets = [], []
    fear_in_non_offensive_dates, num_fear_in_non_offensive_tweets = [], []
    fear_in_offensive_dates_2020, num_fear_in_offensive_tweets_2020 = [], []
    fear_in_non_offensive_dates_2020, num_fear_in_non_offensive_tweets_2020 = [], []
    fear_in_offensive_dates_2021, num_fear_in_offensive_tweets_2021 = [], []
    fear_in_non_offensive_dates_2021, num_fear_in_non_offensive_tweets_2021 = [], []
    fear_in_offensive_dates_2022, num_fear_in_offensive_tweets_2022 = [], []
    fear_in_non_offensive_dates_2022, num_fear_in_non_offensive_tweets_2022 = [], []
    
    for date, num_tweets in fear_in_offensive:
        fear_in_offensive_dates.append(np.datetime64(date))
        num_fear_in_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            fear_in_offensive_dates_2020.append(np.datetime64(date))
            num_fear_in_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            fear_in_offensive_dates_2021.append(np.datetime64(date))
            num_fear_in_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            fear_in_offensive_dates_2022.append(np.datetime64(date))
            num_fear_in_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
    
    for date, num_tweets in fear_in_non_offensive:
        fear_in_non_offensive_dates.append(np.datetime64(date))
        num_fear_in_non_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            fear_in_non_offensive_dates_2020.append(np.datetime64(date))
            num_fear_in_non_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            fear_in_non_offensive_dates_2021.append(np.datetime64(date))
            num_fear_in_non_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            fear_in_non_offensive_dates_2022.append(np.datetime64(date))
            num_fear_in_non_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
            
    # Joy emotion
    joy_in_offensive = sorted(dates_joy_offensive_map.items())
    joy_in_non_offensive = sorted(dates_joy_non_offensive_map.items())
    joy_in_offensive_dates, num_joy_in_offensive_tweets = [], []
    joy_in_non_offensive_dates, num_joy_in_non_offensive_tweets = [], []
    joy_in_offensive_dates_2020, num_joy_in_offensive_tweets_2020 = [], []
    joy_in_non_offensive_dates_2020, num_joy_in_non_offensive_tweets_2020 = [], []
    joy_in_offensive_dates_2021, num_joy_in_offensive_tweets_2021 = [], []
    joy_in_non_offensive_dates_2021, num_joy_in_non_offensive_tweets_2021 = [], []
    joy_in_offensive_dates_2022, num_joy_in_offensive_tweets_2022 = [], []
    joy_in_non_offensive_dates_2022, num_joy_in_non_offensive_tweets_2022 = [], []
    
    for date, num_tweets in joy_in_offensive:
        joy_in_offensive_dates.append(np.datetime64(date))
        num_joy_in_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            joy_in_offensive_dates_2020.append(np.datetime64(date))
            num_joy_in_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            joy_in_offensive_dates_2021.append(np.datetime64(date))
            num_joy_in_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            joy_in_offensive_dates_2022.append(np.datetime64(date))
            num_joy_in_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
            
    for date, num_tweets in joy_in_non_offensive:
        joy_in_non_offensive_dates.append(np.datetime64(date))
        num_joy_in_non_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            joy_in_non_offensive_dates_2020.append(np.datetime64(date))
            num_joy_in_non_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            joy_in_non_offensive_dates_2021.append(np.datetime64(date))
            num_joy_in_non_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            joy_in_non_offensive_dates_2022.append(np.datetime64(date))
            num_joy_in_non_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
            
    # Optimism emotion
    optimism_in_offensive = sorted(dates_optimism_offensive_map.items())
    optimism_in_non_offensive = sorted(dates_optimism_non_offensive_map.items())
    optimism_in_offensive_dates, num_optimism_in_offensive_tweets = [], []
    optimism_in_non_offensive_dates, num_optimism_in_non_offensive_tweets = [], []
    optimism_in_offensive_dates_2020, num_optimism_in_offensive_tweets_2020 = [], []
    optimism_in_non_offensive_dates_2020, num_optimism_in_non_offensive_tweets_2020 = [], []
    optimism_in_offensive_dates_2021, num_optimism_in_offensive_tweets_2021 = [], []
    optimism_in_non_offensive_dates_2021, num_optimism_in_non_offensive_tweets_2021 = [], []
    optimism_in_offensive_dates_2022, num_optimism_in_offensive_tweets_2022 = [], []
    optimism_in_non_offensive_dates_2022, num_optimism_in_non_offensive_tweets_2022 = [], []
    
    for date, num_tweets in optimism_in_offensive:
        optimism_in_offensive_dates.append(np.datetime64(date))
        num_optimism_in_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            optimism_in_offensive_dates_2020.append(np.datetime64(date))
            num_optimism_in_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            optimism_in_offensive_dates_2021.append(np.datetime64(date))
            num_optimism_in_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            optimism_in_offensive_dates_2022.append(np.datetime64(date))
            num_optimism_in_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
    
    for date, num_tweets in optimism_in_non_offensive:
        optimism_in_non_offensive_dates.append(np.datetime64(date))
        num_optimism_in_non_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            optimism_in_non_offensive_dates_2020.append(np.datetime64(date))
            num_optimism_in_non_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            optimism_in_non_offensive_dates_2021.append(np.datetime64(date))
            num_optimism_in_non_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            optimism_in_non_offensive_dates_2022.append(np.datetime64(date))
            num_optimism_in_non_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)


    # Daily # offensive and non-offensive
    dates_offensive_series = pd.Series(dates_offensive_map, index=np.datetime_as_string(offensive_dates))
    dates_non_offensive_series = pd.Series(dates_non_offensive_map, index=np.datetime_as_string(non_offensive_dates))
    
    # emotions with weekly smoothing
    dates_anger_offensive_series = pd.Series(dates_anger_offensive_map, index=np.datetime_as_string(anger_in_offensive_dates))
    dates_anger_offensive_series_rolling = dates_anger_offensive_series.rolling(window=7)
    dates_anger_offensive_series_rolling_mean = dates_anger_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    anger_offensive_dates = dates_anger_offensive_series_rolling_mean.index.to_list()
    anger_offensive_dates = [np.datetime64(date) for date in anger_offensive_dates]
    num_anger_offensive = dates_anger_offensive_series_rolling_mean.to_list()
    
    dates_anger_non_offensive_series = pd.Series(dates_anger_non_offensive_map, index=np.datetime_as_string(anger_in_non_offensive_dates))
    dates_anger_non_offensive_series_rolling = dates_anger_non_offensive_series.rolling(window=7)
    dates_anger_non_offensive_series_rolling_mean = dates_anger_non_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    anger_non_offensive_dates = dates_anger_non_offensive_series_rolling_mean.index.to_list()
    anger_non_offensive_dates = [np.datetime64(date) for date in anger_non_offensive_dates]
    num_anger_non_offensive = dates_anger_non_offensive_series_rolling_mean.to_list()
    
    dates_disgust_offensive_series = pd.Series(dates_disgust_offensive_map, index=np.datetime_as_string(disgust_in_offensive_dates))
    dates_disgust_offensive_series_rolling = dates_disgust_offensive_series.rolling(window=7)
    dates_disgust_offensive_series_rolling_mean = dates_disgust_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    disgust_offensive_dates = dates_disgust_offensive_series_rolling_mean.index.to_list()
    disgust_offensive_dates = [np.datetime64(date) for date in disgust_offensive_dates]
    num_disgust_offensive = dates_disgust_offensive_series_rolling_mean.to_list()
    
    dates_disgust_non_offensive_series = pd.Series(dates_disgust_non_offensive_map, index=np.datetime_as_string(disgust_in_non_offensive_dates))
    dates_disgust_non_offensive_series_rolling = dates_disgust_non_offensive_series.rolling(window=7)
    dates_disgust_non_offensive_series_rolling_mean = dates_disgust_non_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    disgust_non_offensive_dates = dates_disgust_non_offensive_series_rolling_mean.index.to_list()
    disgust_non_offensive_dates = [np.datetime64(date) for date in disgust_non_offensive_dates]
    num_disgust_non_offensive = dates_disgust_non_offensive_series_rolling_mean.to_list()
    
    dates_fear_offensive_series = pd.Series(dates_fear_offensive_map, index=np.datetime_as_string(fear_in_offensive_dates))
    dates_fear_offensive_series_rolling = dates_fear_offensive_series.rolling(window=7)
    dates_fear_offensive_series_rolling_mean = dates_fear_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    fear_offensive_dates = dates_fear_offensive_series_rolling_mean.index.to_list()
    fear_offensive_dates = [np.datetime64(date) for date in fear_offensive_dates]
    num_fear_offensive = dates_fear_offensive_series_rolling_mean.to_list()
    
    dates_fear_non_offensive_series = pd.Series(dates_fear_non_offensive_map, index=np.datetime_as_string(fear_in_non_offensive_dates))
    dates_fear_non_offensive_series_rolling = dates_fear_non_offensive_series.rolling(window=7)
    dates_fear_non_offensive_series_rolling_mean = dates_fear_non_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    fear_non_offensive_dates = dates_fear_non_offensive_series_rolling_mean.index.to_list()
    fear_non_offensive_dates = [np.datetime64(date) for date in fear_non_offensive_dates]
    num_fear_non_offensive = dates_fear_non_offensive_series_rolling_mean.to_list()
    
    dates_joy_offensive_series = pd.Series(dates_joy_offensive_map, index=np.datetime_as_string(joy_in_offensive_dates))
    dates_joy_offensive_series_rolling = dates_joy_offensive_series.rolling(window=7)
    dates_joy_offensive_series_rolling_mean = dates_joy_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    joy_offensive_dates = dates_joy_offensive_series_rolling_mean.index.to_list()
    joy_offensive_dates = [np.datetime64(date) for date in joy_offensive_dates]
    num_joy_offensive = dates_joy_offensive_series_rolling_mean.to_list()
    
    dates_joy_non_offensive_series = pd.Series(dates_joy_non_offensive_map, index=np.datetime_as_string(joy_in_non_offensive_dates))
    dates_joy_non_offensive_series_rolling = dates_joy_non_offensive_series.rolling(window=7)
    dates_joy_non_offensive_series_rolling_mean = dates_joy_non_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    joy_non_offensive_dates = dates_joy_non_offensive_series_rolling_mean.index.to_list()
    joy_non_offensive_dates = [np.datetime64(date) for date in joy_non_offensive_dates]
    num_joy_non_offensive = dates_joy_non_offensive_series_rolling_mean.to_list()
    
    dates_optimism_offensive_series = pd.Series(dates_optimism_offensive_map, index=np.datetime_as_string(optimism_in_offensive_dates))
    dates_optimism_offensive_series_rolling = dates_optimism_offensive_series.rolling(window=7)
    dates_optimism_offensive_series_rolling_mean = dates_optimism_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    optimism_offensive_dates = dates_optimism_offensive_series_rolling_mean.index.to_list()
    optimism_offensive_dates = [np.datetime64(date) for date in optimism_offensive_dates]
    num_optimism_offensive = dates_optimism_offensive_series_rolling_mean.to_list()
    
    dates_optimism_non_offensive_series = pd.Series(dates_optimism_non_offensive_map, index=np.datetime_as_string(optimism_in_non_offensive_dates))
    dates_optimism_non_offensive_series_rolling = dates_optimism_non_offensive_series.rolling(window=7)
    dates_optimism_non_offensive_series_rolling_mean = dates_optimism_non_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    optimism_non_offensive_dates = dates_optimism_non_offensive_series_rolling_mean.index.to_list()
    optimism_non_offensive_dates = [np.datetime64(date) for date in optimism_non_offensive_dates]
    num_optimism_non_offensive = dates_optimism_non_offensive_series_rolling_mean.to_list()

    # Vertical lines. The line positions are obtained by first running change point detection algorithm on data
    events_lines = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']
    daily_tweets_vlines = [np.datetime64('2020-05-31'), np.datetime64('2020-08-24'), np.datetime64('2020-09-23'), np.datetime64('2021-01-07'), np.datetime64('2021-04-22'), np.datetime64('2021-05-27'), np.datetime64('2021-11-23'), np.datetime64('2022-05-28')]
    emotions_in_offensive_tweets_vlines = [np.datetime64('2020-05-31'), np.datetime64('2020-08-24'), np.datetime64('2021-01-02'), np.datetime64('2021-04-17'), np.datetime64('2021-11-23'), np.datetime64('2022-05-23')] #2021-11-23 did not appear in optimism
    emotions_in_non_offensive_tweets_vlines = [np.datetime64('2020-05-31'), np.datetime64('2020-08-24'), np.datetime64('2020-09-18'), np.datetime64('2021-01-02'), np.datetime64('2021-04-17'), np.datetime64('2021-11-23'), np.datetime64('2022-02-22'), np.datetime64('2022-05-23')] #np.datetime64('2022-10-15')
    
    # Plot Daily tweets, % of emotions wrt offensive tweets, % of emotions wrt non-offensive tweets,
    month_year_formatter = mdates.DateFormatter('%b %Y')
    labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Optimism']
    x = np.arange(len(labels))
    ncols = 3
    nrows = 1
    anger_line, disgust_line, fear_line, joy_line, optimism_line = None, None, None, None, None
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(14, 3.5), layout="constrained")#12, 2.7
    # add an artist, in this case a nice label in the middle...
    for col in range(ncols):
        if col == 0:
            # daily tweets
            axs[col].xaxis.set_major_formatter(month_year_formatter)
            for label in axs[col].get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
            axs[col].plot(non_offensive_dates, num_non_offensive_tweets, label='non-offensive', linewidth=1)
            axs[col].plot(offensive_dates, num_offensive_tweets, label='offensive', linewidth=1)
            # plot vertical line
            for i, date in enumerate(daily_tweets_vlines):
                axs[col].axvline(date, ls='--', color='#929591')
                if i == 1:
                    axs[col].text(np.datetime64('2020-08-01'), 1250000, events_lines[i], fontsize=8)
                elif i == 4:
                    axs[col].text(np.datetime64('2021-04-01'), 1250000, events_lines[i], fontsize=8)
                else:
                    axs[col].text(date, 1250000, events_lines[i], fontsize=8)
            axs[col].set_ylabel('# of Tweets')
            axs[col].set_yscale('log')
            axs[col].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*1:.0f}'))
            axs[col].set_title("A) Number of Daily Tweets", loc='left', fontsize=10, fontweight='medium', pad=15)
            # Shrink current axis's height by 10% on the bottom
            box = axs[col].get_position()
            #axs[col].legend(ncols=2, loc="lower right")
            axs[col].legend(ncols=2, loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.2))
            axs[col].grid(True)

        elif col == 1:
            # Emotions of offensive tweets 
            axs[col].xaxis.set_major_formatter(month_year_formatter)
            for label in axs[col].get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
            axs[col].plot(anger_offensive_dates, num_anger_offensive, label='Anger', linewidth=1)
            axs[col].plot(disgust_offensive_dates, num_disgust_offensive, label='Disgust', linewidth=1)
            axs[col].plot(fear_offensive_dates, num_fear_offensive, label='Fear', linewidth=1)
            axs[col].plot(joy_offensive_dates, num_joy_offensive, label='Joy', linewidth=1)
            axs[col].plot(optimism_offensive_dates, num_optimism_offensive, label='Optimism', linewidth=1)
            
            # plot vertical line
            for i, date in enumerate(emotions_in_offensive_tweets_vlines):
                axs[col].axvline(date, ls='--', color='#929591')
                axs[col].text(date, 0.48, events_lines[i], fontsize=8, va="center", ha="center")
            axs[col].set_ylabel('% of Tweets')
            axs[col].set_yscale('log')
            #axs[col].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*100:.2f}'))
            axs[col].yaxis.set_major_formatter(PercentFormatter())
            axs[col].set_title("B) Proportion of Offensive Tweets by Emotion", loc='left', fontsize=10, fontweight='medium', pad=15)
            #axs[col].legend()
            axs[col].grid(True)

        elif col == 2:
            # Emotions of non-offensive tweets
            axs[col].xaxis.set_major_formatter(month_year_formatter)
            for label in axs[col].get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
            anger_line = axs[col].plot(anger_non_offensive_dates, num_anger_non_offensive, label='Anger', linewidth=1)
            disgust_line = axs[col].plot(disgust_non_offensive_dates, num_disgust_non_offensive, label='Disgust', linewidth=1)
            fear_line = axs[col].plot(fear_non_offensive_dates, num_fear_non_offensive, label='Fear', linewidth=1)
            joy_line = axs[col].plot(joy_non_offensive_dates, num_joy_non_offensive, label='Joy', linewidth=1)
            optimism_line = axs[col].plot(optimism_non_offensive_dates, num_optimism_non_offensive, label='Optimism', linewidth=1)

            # plot vertical line
            for i, date in enumerate(emotions_in_non_offensive_tweets_vlines):
                axs[col].axvline(date, ls='--', color='#929591')
                if i == 1:
                    axs[col].text(np.datetime64('2020-08-01'), 2.2, events_lines[i], fontsize=8)
                else:
                    axs[col].text(date, 2.2, events_lines[i], fontsize=8)
            axs[col].set_ylabel('% of Tweets')
            axs[col].set_yscale('log')
            #axs[col].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
            axs[col].yaxis.set_major_formatter(PercentFormatter())
            axs[col].set_title("C) Proportion of Non-offensive Tweets by Emotion", loc='left', fontsize=10, fontweight='medium', pad=15)
            axs[col].grid(True)
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(default_colors, labels)]
    lines = [anger_line, disgust_line, fear_line, joy_line, optimism_line]
    fig.legend(labels=labels, loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.7, 0.1), handles=patches, handlelength=1.0)
    try:
        fig.savefig('figures/num_daily_tweets_and_prop_emotions_full_non_offensive.eps', format='eps', bbox_inches='tight')
    except FileNotFoundError:
        # this allows the script to keep going if run interactively and
        # the directory above doesn't exist
        pass
    
    # Plot bar chart of emotions in 2020, 2021, and 2022 all in one figure
    labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Optimism']
    x = np.arange(len(labels))
    width = 0.35
    ncols = 3
    nrows = 1
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 2.7), layout="constrained")#12, 3
    # add an artist, in this case a nice label in the middle...
    for col in range(3):        
        if col == 0:
            # Bar chart of emotions in 2020
            perc_anger_non_offensive = sum(num_anger_in_non_offensive_tweets_2020)
            perc_anger_offensive = sum(num_anger_in_offensive_tweets_2020)
            perc_disgust_non_offensive = sum(num_disgust_in_non_offensive_tweets_2020)
            perc_disgust_offensive = sum(num_disgust_in_offensive_tweets_2020)
            perc_fear_non_offensive = sum(num_fear_in_non_offensive_tweets_2020)
            perc_fear_offensive = sum(num_fear_in_offensive_tweets_2020)
            perc_joy_non_offensive = sum(num_joy_in_non_offensive_tweets_2020)
            perc_joy_offensive = sum(num_joy_in_offensive_tweets_2020)
            perc_optimism_non_offensive = sum(num_optimism_in_non_offensive_tweets_2020)
            perc_optimism_offensive = sum(num_optimism_in_offensive_tweets_2020)

            all_non_offensive = [
                round(perc_anger_non_offensive,2), 
                round(perc_disgust_non_offensive,2), 
                round(perc_fear_non_offensive,2), 
                round(perc_joy_non_offensive,2), 
                round(perc_optimism_non_offensive,2),
                                ]

            all_offensive = [
                round(perc_anger_offensive,2), 
                round(perc_disgust_offensive,2), 
                round(perc_fear_offensive,2), 
                round(perc_joy_offensive,2), 
                round(perc_optimism_offensive,2),
                                ]

            rects1 = axs[col].bar(x - width/2, all_non_offensive, width, label='Non-offensive')
            rects2 = axs[col].bar(x + width/2, all_offensive, width, label='Offensive')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            axs[col].set_ylabel('% of Tweets')
            axs[col].set_title('D) 2020 Tweets Emotional Distribution', loc='left', fontsize=10, fontweight='medium')
            axs[col].set_xticks(x, labels)
            for label in axs[col].get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
            #axs[row, col].legend()
            #axs[row, col].bar_label(rects1, padding=3)
            #axs[row, col].bar_label(rects2, padding=3)

        elif col == 1:
            # Bar chart of emotions in 2021
            perc_anger_non_offensive = sum(num_anger_in_non_offensive_tweets_2021)
            perc_anger_offensive = sum(num_anger_in_offensive_tweets_2021)
            perc_disgust_non_offensive = sum(num_disgust_in_non_offensive_tweets_2021)
            perc_disgust_offensive = sum(num_disgust_in_offensive_tweets_2021)
            perc_fear_non_offensive = sum(num_fear_in_non_offensive_tweets_2021)
            perc_fear_offensive = sum(num_fear_in_offensive_tweets_2021)
            perc_joy_non_offensive = sum(num_joy_in_non_offensive_tweets_2021)
            perc_joy_offensive = sum(num_joy_in_offensive_tweets_2021)
            perc_optimism_non_offensive = sum(num_optimism_in_non_offensive_tweets_2021)
            perc_optimism_offensive = sum(num_optimism_in_offensive_tweets_2021)

            all_non_offensive = [
                round(perc_anger_non_offensive,2), 
                round(perc_disgust_non_offensive,2), 
                round(perc_fear_non_offensive,2), 
                round(perc_joy_non_offensive,2), 
                round(perc_optimism_non_offensive,2),
                                ]

            all_offensive = [
                round(perc_anger_offensive,2), 
                round(perc_disgust_offensive,2), 
                round(perc_fear_offensive,2), 
                round(perc_joy_offensive,2), 
                round(perc_optimism_offensive,2),
                                ]

            rects1 = axs[col].bar(x - width/2, all_non_offensive, width, label='Non-offensive')
            rects2 = axs[col].bar(x + width/2, all_offensive, width, label='Offensive')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            axs[col].set_ylabel('% of Tweets')
            axs[col].set_title('E) 2021 Tweets Emotional Distribution', loc='left', fontsize=10, fontweight='medium')
            axs[col].set_xticks(x, labels)
            for label in axs[col].get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
            #axs[row, col].legend()
            #axs[row, col].bar_label(rects1, padding=3)
            #axs[row, col].bar_label(rects2, padding=3)

        elif col == 2:
            # Bar chart of emotions in 2022
            perc_anger_non_offensive = sum(num_anger_in_non_offensive_tweets_2022)
            perc_anger_offensive = sum(num_anger_in_offensive_tweets_2022)
            perc_disgust_non_offensive = sum(num_disgust_in_non_offensive_tweets_2022)
            perc_disgust_offensive = sum(num_disgust_in_offensive_tweets_2022)
            perc_fear_non_offensive = sum(num_fear_in_non_offensive_tweets_2022)
            perc_fear_offensive = sum(num_fear_in_offensive_tweets_2022)
            perc_joy_non_offensive = sum(num_joy_in_non_offensive_tweets_2022)
            perc_joy_offensive = sum(num_joy_in_offensive_tweets_2022)
            perc_optimism_non_offensive = sum(num_optimism_in_non_offensive_tweets_2022)
            perc_optimism_offensive = sum(num_optimism_in_offensive_tweets_2022)

            all_non_offensive = [
                round(perc_anger_non_offensive,2), 
                round(perc_disgust_non_offensive,2), 
                round(perc_fear_non_offensive,2), 
                round(perc_joy_non_offensive,2), 
                round(perc_optimism_non_offensive,2),
                                ]

            all_offensive = [
                round(perc_anger_offensive,2), 
                round(perc_disgust_offensive,2), 
                round(perc_fear_offensive,2), 
                round(perc_joy_offensive,2),
                round(perc_optimism_offensive,2)
                                ]

            rects1 = axs[col].bar(x - width/2, all_non_offensive, width, label='Non-offensive')
            rects2 = axs[col].bar(x + width/2, all_offensive, width, label='Offensive')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            axs[col].set_ylabel('% of Tweets')
            axs[col].set_title('F) 2022 Tweets Emotional Distribution', loc='left', fontsize=10, fontweight='medium')
            axs[col].set_xticks(x, labels)
            for label in axs[col].get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
            #axs[col].legend()
            #axs[row, col].bar_label(rects1, padding=3)
            #axs[row, col].bar_label(rects2, padding=3)
    
    labels = ['Non-offensive', 'offensive']
    fig.legend(labels=labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.52, 0.03))
    plt.savefig('figures/emotion_distributions_by_year_full_non_offensive.eps', format='eps', bbox_inches='tight')
    
    
def plot_emotions_offenders_receivers(datasets, 
                                      save_as, 
                                      events_lines,
                                      emotions_in_offensive_tweets_vlines,
                                      emotions_in_non_offensive_tweets_vlines):
    '''
    Plot the number of tweets vs date of the offensive and non-offensive tweets 
    in out datasets, proportion of offensive tweets by emotions, proportion of non-offensive tweets by emotions,
    bar chart of emotion distribution. 
    Args:
        datasets (List[String]): Contains the dataset path of each year
    Returns:
        None
    '''
        
    num_offensive, num_non_offensive = 0, 0
    num_offensive_2020, num_non_offensive_2020 = 0, 0
    num_offensive_2021, num_non_offensive_2021 = 0, 0
    num_offensive_2022, num_non_offensive_2022 = 0, 0
    # Daily tweets 
    dates_offensive_map = {}
    dates_non_offensive_map = {}
    # emotions 
    dates_anger_offensive_map, dates_anger_non_offensive_map = {}, {}
    dates_disgust_offensive_map, dates_disgust_non_offensive_map = {}, {}
    dates_fear_offensive_map, dates_fear_non_offensive_map = {}, {}
    dates_joy_offensive_map, dates_joy_non_offensive_map = {}, {}
    #dates_love_offensive_map, dates_love_non_offensive_map = {}, {}
    dates_optimism_offensive_map, dates_optimism_non_offensive_map = {}, {}
    #dates_sadness_offensive_map, dates_sadness_non_offensive_map = {}, {}
    
    for i, file_path in enumerate(datasets):
        with open(file_path) as csv_file_path_handle:
            csv_reader = csv.reader(csv_file_path_handle, delimiter=',')
            for j, row in enumerate(csv_reader):
                date = row[3]
                anger = row[9]
                disgust = row[11]
                fear = row[12]
                joy = row[13]
                #love = row[14]
                optimism = row[15]
                #sadness = row[17]
                label = row[20]
                if label == '1':
                    num_offensive += 1
                    date_list = date.split('-')
                    year = date_list[0]
                    if year == '2020':
                        num_offensive_2020 += 1
                    elif year == '2021':
                        num_offensive_2021 += 1
                    elif year == '2022':
                        num_offensive_2022 += 1
                        
                    # Daily tweets
                    if date in dates_offensive_map:
                        dates_offensive_map[date] += 1
                    else:
                        dates_offensive_map[date] = 1
                    # Anger emotion
                    if anger == '1':
                        if date in dates_anger_offensive_map:
                            dates_anger_offensive_map[date] += 1
                        else:
                            dates_anger_offensive_map[date] = 1
                    # Disgust emotion
                    if disgust == '1':
                        if date in dates_disgust_offensive_map:
                            dates_disgust_offensive_map[date] += 1
                        else:
                            dates_disgust_offensive_map[date] = 1
                    # Fear emotion
                    if fear == '1':
                        if date in dates_fear_offensive_map:
                            dates_fear_offensive_map[date] += 1
                        else:
                            dates_fear_offensive_map[date] = 1
                    # Joy emotion       
                    if joy == '1':
                        if date in dates_joy_offensive_map:
                            dates_joy_offensive_map[date] += 1
                        else:
                            dates_joy_offensive_map[date] = 1
                    # Optimism emotion
                    if optimism == '1':
                        if date in dates_optimism_offensive_map:
                            dates_optimism_offensive_map[date] += 1
                        else:
                            dates_optimism_offensive_map[date] = 1
                            
                elif label == '0':
                    num_non_offensive += 1
                    date_list = date.split('-')
                    year = date_list[0]
                    if year == '2020':
                        num_non_offensive_2020 +=1
                    elif year == '2021':
                        num_non_offensive_2021 += 1
                    elif year == '2022':
                        num_non_offensive_2022 += 1
                    
                    # Daily tweets
                    if date in dates_non_offensive_map:
                        dates_non_offensive_map[date] += 1
                    else:
                        dates_non_offensive_map[date] = 1
                    # Anger emotion
                    if anger == '1':
                        if date in dates_anger_non_offensive_map:
                            dates_anger_non_offensive_map[date] += 1
                        else:
                            dates_anger_non_offensive_map[date] = 1
                    # Disgust emotion
                    if disgust == '1':
                        if date in dates_disgust_non_offensive_map:
                            dates_disgust_non_offensive_map[date] += 1
                        else:
                            dates_disgust_non_offensive_map[date] = 1
                    # Fear emotion
                    if fear == '1':
                        if date in dates_fear_non_offensive_map:
                            dates_fear_non_offensive_map[date] += 1
                        else:
                            dates_fear_non_offensive_map[date] = 1
                    # Joy emotion       
                    if joy == '1':
                        if date in dates_joy_non_offensive_map:
                            dates_joy_non_offensive_map[date] += 1
                        else:
                            dates_joy_non_offensive_map[date] = 1
                    # Optimism emotion
                    if optimism == '1':
                        if date in dates_optimism_non_offensive_map:
                            dates_optimism_non_offensive_map[date] += 1
                        else:
                            dates_optimism_non_offensive_map[date] = 1
    
    print(f'Number of overall offensive: {num_offensive}, overall non-offensive: {num_non_offensive}')
    print(f'Number of offensive in 2020: {num_offensive_2020}, non-offensive in 2020: {num_non_offensive_2020}')
    print(f'Number of offensive in 2021: {num_offensive_2021}, non-offensive in 2021: {num_non_offensive_2021}')
    print(f'Number of offensive in 2022: {num_offensive_2022}, non-offensive in 2022: {num_non_offensive_2022}')
    print()
    
    # Daily tweets
    offensive = sorted(dates_offensive_map.items())
    non_offensive = sorted(dates_non_offensive_map.items())
    offensive_dates, num_offensive_tweets = [np.datetime64(date) for date, _ in offensive], [num_tweets for _, num_tweets in offensive]
    non_offensive_dates, num_non_offensive_tweets = [np.datetime64(date) for date, _ in non_offensive], [num_tweets for _, num_tweets in non_offensive]
    print(f'Offensive dates: {offensive_dates[:5]}')
    print(f'# offensive dates: {num_offensive_tweets[:5]}')
    print(f'Non-offensive dates: {non_offensive_dates[:5]}')
    print(f'# non-offensive dates: {num_non_offensive_tweets[:5]}')
    print()
    
    # Anger emotions
    anger_in_offensive = sorted(dates_anger_offensive_map.items())
    anger_in_non_offensive = sorted(dates_anger_non_offensive_map.items())
    anger_in_offensive_dates, num_anger_in_offensive_tweets = [], []
    anger_in_non_offensive_dates, num_anger_in_non_offensive_tweets = [], []
    anger_in_offensive_dates_2020, num_anger_in_offensive_tweets_2020 = [], []
    anger_in_non_offensive_dates_2020, num_anger_in_non_offensive_tweets_2020 = [], []
    anger_in_offensive_dates_2021, num_anger_in_offensive_tweets_2021 = [], []
    anger_in_non_offensive_dates_2021, num_anger_in_non_offensive_tweets_2021 = [], []
    anger_in_offensive_dates_2022, num_anger_in_offensive_tweets_2022 = [], []
    anger_in_non_offensive_dates_2022, num_anger_in_non_offensive_tweets_2022 = [], []
    
    for date, num_tweets in anger_in_offensive:
        anger_in_offensive_dates.append(np.datetime64(date))
        num_anger_in_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            anger_in_offensive_dates_2020.append(np.datetime64(date))
            num_anger_in_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            anger_in_offensive_dates_2021.append(np.datetime64(date))
            num_anger_in_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            anger_in_offensive_dates_2022.append(np.datetime64(date))
            num_anger_in_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
            
    for date, num_tweets in anger_in_non_offensive:
        anger_in_non_offensive_dates.append(np.datetime64(date))
        num_anger_in_non_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            anger_in_non_offensive_dates_2020.append(np.datetime64(date))
            num_anger_in_non_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            anger_in_non_offensive_dates_2021.append(np.datetime64(date))
            num_anger_in_non_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            anger_in_non_offensive_dates_2022.append(np.datetime64(date))
            num_anger_in_non_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
        
    # Disgust emotion
    disgust_in_offensive = sorted(dates_disgust_offensive_map.items())
    disgust_in_non_offensive = sorted(dates_disgust_non_offensive_map.items())
    disgust_in_offensive_dates, num_disgust_in_offensive_tweets = [], []
    disgust_in_non_offensive_dates, num_disgust_in_non_offensive_tweets = [], []
    disgust_in_offensive_dates_2020, num_disgust_in_offensive_tweets_2020 = [], []
    disgust_in_non_offensive_dates_2020, num_disgust_in_non_offensive_tweets_2020 = [], []
    disgust_in_offensive_dates_2021, num_disgust_in_offensive_tweets_2021 = [], []
    disgust_in_non_offensive_dates_2021, num_disgust_in_non_offensive_tweets_2021 = [], []
    disgust_in_offensive_dates_2022, num_disgust_in_offensive_tweets_2022 = [], []
    disgust_in_non_offensive_dates_2022, num_disgust_in_non_offensive_tweets_2022 = [], []
    
    for date, num_tweets in disgust_in_offensive:
        disgust_in_offensive_dates.append(np.datetime64(date))
        num_disgust_in_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            disgust_in_offensive_dates_2020.append(np.datetime64(date))
            num_disgust_in_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            disgust_in_offensive_dates_2021.append(np.datetime64(date))
            num_disgust_in_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            disgust_in_offensive_dates_2022.append(np.datetime64(date))
            num_disgust_in_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
           
    for date, num_tweets in disgust_in_non_offensive:
        disgust_in_non_offensive_dates.append(np.datetime64(date))
        num_disgust_in_non_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            disgust_in_non_offensive_dates_2020.append(np.datetime64(date))
            num_disgust_in_non_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            disgust_in_non_offensive_dates_2021.append(np.datetime64(date))
            num_disgust_in_non_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            disgust_in_non_offensive_dates_2022.append(np.datetime64(date))
            num_disgust_in_non_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
            
    # Fear emotion
    fear_in_offensive = sorted(dates_fear_offensive_map.items())
    fear_in_non_offensive = sorted(dates_fear_non_offensive_map.items())
    fear_in_offensive_dates, num_fear_in_offensive_tweets = [], []
    fear_in_non_offensive_dates, num_fear_in_non_offensive_tweets = [], []
    fear_in_offensive_dates_2020, num_fear_in_offensive_tweets_2020 = [], []
    fear_in_non_offensive_dates_2020, num_fear_in_non_offensive_tweets_2020 = [], []
    fear_in_offensive_dates_2021, num_fear_in_offensive_tweets_2021 = [], []
    fear_in_non_offensive_dates_2021, num_fear_in_non_offensive_tweets_2021 = [], []
    fear_in_offensive_dates_2022, num_fear_in_offensive_tweets_2022 = [], []
    fear_in_non_offensive_dates_2022, num_fear_in_non_offensive_tweets_2022 = [], []
    
    for date, num_tweets in fear_in_offensive:
        fear_in_offensive_dates.append(np.datetime64(date))
        num_fear_in_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            fear_in_offensive_dates_2020.append(np.datetime64(date))
            num_fear_in_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            fear_in_offensive_dates_2021.append(np.datetime64(date))
            num_fear_in_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            fear_in_offensive_dates_2022.append(np.datetime64(date))
            num_fear_in_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
    
    for date, num_tweets in fear_in_non_offensive:
        fear_in_non_offensive_dates.append(np.datetime64(date))
        num_fear_in_non_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            fear_in_non_offensive_dates_2020.append(np.datetime64(date))
            num_fear_in_non_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            fear_in_non_offensive_dates_2021.append(np.datetime64(date))
            num_fear_in_non_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            fear_in_non_offensive_dates_2022.append(np.datetime64(date))
            num_fear_in_non_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
            
    # Joy emotion
    joy_in_offensive = sorted(dates_joy_offensive_map.items())
    joy_in_non_offensive = sorted(dates_joy_non_offensive_map.items())
    joy_in_offensive_dates, num_joy_in_offensive_tweets = [], []
    joy_in_non_offensive_dates, num_joy_in_non_offensive_tweets = [], []
    joy_in_offensive_dates_2020, num_joy_in_offensive_tweets_2020 = [], []
    joy_in_non_offensive_dates_2020, num_joy_in_non_offensive_tweets_2020 = [], []
    joy_in_offensive_dates_2021, num_joy_in_offensive_tweets_2021 = [], []
    joy_in_non_offensive_dates_2021, num_joy_in_non_offensive_tweets_2021 = [], []
    joy_in_offensive_dates_2022, num_joy_in_offensive_tweets_2022 = [], []
    joy_in_non_offensive_dates_2022, num_joy_in_non_offensive_tweets_2022 = [], []
    
    for date, num_tweets in joy_in_offensive:
        joy_in_offensive_dates.append(np.datetime64(date))
        num_joy_in_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            joy_in_offensive_dates_2020.append(np.datetime64(date))
            num_joy_in_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            joy_in_offensive_dates_2021.append(np.datetime64(date))
            num_joy_in_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            joy_in_offensive_dates_2022.append(np.datetime64(date))
            num_joy_in_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
            
    for date, num_tweets in joy_in_non_offensive:
        joy_in_non_offensive_dates.append(np.datetime64(date))
        num_joy_in_non_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            joy_in_non_offensive_dates_2020.append(np.datetime64(date))
            num_joy_in_non_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            joy_in_non_offensive_dates_2021.append(np.datetime64(date))
            num_joy_in_non_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            joy_in_non_offensive_dates_2022.append(np.datetime64(date))
            num_joy_in_non_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
            
    # Optimism emotion
    optimism_in_offensive = sorted(dates_optimism_offensive_map.items())
    optimism_in_non_offensive = sorted(dates_optimism_non_offensive_map.items())
    optimism_in_offensive_dates, num_optimism_in_offensive_tweets = [], []
    optimism_in_non_offensive_dates, num_optimism_in_non_offensive_tweets = [], []
    optimism_in_offensive_dates_2020, num_optimism_in_offensive_tweets_2020 = [], []
    optimism_in_non_offensive_dates_2020, num_optimism_in_non_offensive_tweets_2020 = [], []
    optimism_in_offensive_dates_2021, num_optimism_in_offensive_tweets_2021 = [], []
    optimism_in_non_offensive_dates_2021, num_optimism_in_non_offensive_tweets_2021 = [], []
    optimism_in_offensive_dates_2022, num_optimism_in_offensive_tweets_2022 = [], []
    optimism_in_non_offensive_dates_2022, num_optimism_in_non_offensive_tweets_2022 = [], []
    
    for date, num_tweets in optimism_in_offensive:
        optimism_in_offensive_dates.append(np.datetime64(date))
        num_optimism_in_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            optimism_in_offensive_dates_2020.append(np.datetime64(date))
            num_optimism_in_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            optimism_in_offensive_dates_2021.append(np.datetime64(date))
            num_optimism_in_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            optimism_in_offensive_dates_2022.append(np.datetime64(date))
            num_optimism_in_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
    
    for date, num_tweets in optimism_in_non_offensive:
        optimism_in_non_offensive_dates.append(np.datetime64(date))
        num_optimism_in_non_offensive_tweets.append((num_tweets/(num_offensive+num_non_offensive))*100)
        year = date.split('-')[0]
        if year == '2020':
            optimism_in_non_offensive_dates_2020.append(np.datetime64(date))
            num_optimism_in_non_offensive_tweets_2020.append((num_tweets/(num_offensive_2020+num_non_offensive_2020))*100)
        elif year == '2021':
            optimism_in_non_offensive_dates_2021.append(np.datetime64(date))
            num_optimism_in_non_offensive_tweets_2021.append((num_tweets/(num_offensive_2021+num_non_offensive_2021))*100)
        else:
            optimism_in_non_offensive_dates_2022.append(np.datetime64(date))
            num_optimism_in_non_offensive_tweets_2022.append((num_tweets/(num_offensive_2022+num_non_offensive_2022))*100)
    
    # Write the raw counts to file   
    anger_offensive_dates = [date for date, _ in anger_in_offensive]
    anger_offensive_total_tweets = [total for _, total in anger_in_offensive]
    anger_non_offensive_dates = [date for date, _ in anger_in_non_offensive]
    anger_non_offensive_total_tweets = [total for _, total in anger_in_non_offensive]
    
    disgust_offensive_dates = [date for date, _ in disgust_in_offensive]
    disgust_offensive_total_tweets = [total for _, total in disgust_in_offensive]
    disgust_non_offensive_dates = [date for date, _ in disgust_in_non_offensive]
    disgust_non_offensive_total_tweets = [total for _, total in anger_in_non_offensive]
    
    fear_offensive_dates = [date for date, _ in fear_in_offensive]
    fear_offensive_total_tweets = [total for _, total in fear_in_offensive]
    fear_non_offensive_dates = [date for date, _ in fear_in_non_offensive]
    fear_non_offensive_total_tweets = [total for _, total in fear_in_non_offensive]
    
    joy_offensive_dates = [date for date, _ in joy_in_offensive]
    joy_offensive_total_tweets = [total for _, total in joy_in_offensive]
    joy_non_offensive_dates = [date for date, _ in joy_in_non_offensive]
    joy_non_offensive_total_tweets = [total for _, total in joy_in_non_offensive]
    
    optimism_offensive_dates = [date for date, _ in optimism_in_offensive]
    optimism_offensive_total_tweets = [total for _, total in optimism_in_offensive]
    optimism_non_offensive_dates = [date for date, _ in optimism_in_non_offensive]
    optimism_non_offensive_total_tweets = [total for _, total in optimism_in_non_offensive]
        
    # Daily # offensive and non-offensive
    dates_offensive_series = pd.Series(dates_offensive_map, index=np.datetime_as_string(offensive_dates))
    dates_non_offensive_series = pd.Series(dates_non_offensive_map, index=np.datetime_as_string(non_offensive_dates))
    
    # emotions with smoothing
    dates_anger_offensive_series = pd.Series(dates_anger_offensive_map, index=np.datetime_as_string(anger_in_offensive_dates))
    dates_anger_offensive_series_rolling = dates_anger_offensive_series.rolling(window=7)
    dates_anger_offensive_series_rolling_mean = dates_anger_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    anger_offensive_dates = dates_anger_offensive_series_rolling_mean.index.to_list()
    anger_offensive_dates = [np.datetime64(date) for date in anger_offensive_dates]
    num_anger_offensive = dates_anger_offensive_series_rolling_mean.to_list()
    
    dates_anger_non_offensive_series = pd.Series(dates_anger_non_offensive_map, index=np.datetime_as_string(anger_in_non_offensive_dates))
    dates_anger_non_offensive_series_rolling = dates_anger_non_offensive_series.rolling(window=7)
    dates_anger_non_offensive_series_rolling_mean = dates_anger_non_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    anger_non_offensive_dates = dates_anger_non_offensive_series_rolling_mean.index.to_list()
    anger_non_offensive_dates = [np.datetime64(date) for date in anger_non_offensive_dates]
    num_anger_non_offensive = dates_anger_non_offensive_series_rolling_mean.to_list()
    
    dates_disgust_offensive_series = pd.Series(dates_disgust_offensive_map, index=np.datetime_as_string(disgust_in_offensive_dates))
    dates_disgust_offensive_series_rolling = dates_disgust_offensive_series.rolling(window=7)
    dates_disgust_offensive_series_rolling_mean = dates_disgust_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    disgust_offensive_dates = dates_disgust_offensive_series_rolling_mean.index.to_list()
    disgust_offensive_dates = [np.datetime64(date) for date in disgust_offensive_dates]
    num_disgust_offensive = dates_disgust_offensive_series_rolling_mean.to_list()
    
    dates_disgust_non_offensive_series = pd.Series(dates_disgust_non_offensive_map, index=np.datetime_as_string(disgust_in_non_offensive_dates))
    dates_disgust_non_offensive_series_rolling = dates_disgust_non_offensive_series.rolling(window=7)
    dates_disgust_non_offensive_series_rolling_mean = dates_disgust_non_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    disgust_non_offensive_dates = dates_disgust_non_offensive_series_rolling_mean.index.to_list()
    disgust_non_offensive_dates = [np.datetime64(date) for date in disgust_non_offensive_dates]
    num_disgust_non_offensive = dates_disgust_non_offensive_series_rolling_mean.to_list()
    
    dates_fear_offensive_series = pd.Series(dates_fear_offensive_map, index=np.datetime_as_string(fear_in_offensive_dates))
    dates_fear_offensive_series_rolling = dates_fear_offensive_series.rolling(window=7)
    dates_fear_offensive_series_rolling_mean = dates_fear_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    fear_offensive_dates = dates_fear_offensive_series_rolling_mean.index.to_list()
    fear_offensive_dates = [np.datetime64(date) for date in fear_offensive_dates]
    num_fear_offensive = dates_fear_offensive_series_rolling_mean.to_list()
    
    dates_fear_non_offensive_series = pd.Series(dates_fear_non_offensive_map, index=np.datetime_as_string(fear_in_non_offensive_dates))
    dates_fear_non_offensive_series_rolling = dates_fear_non_offensive_series.rolling(window=7)
    dates_fear_non_offensive_series_rolling_mean = dates_fear_non_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    fear_non_offensive_dates = dates_fear_non_offensive_series_rolling_mean.index.to_list()
    fear_non_offensive_dates = [np.datetime64(date) for date in fear_non_offensive_dates]
    num_fear_non_offensive = dates_fear_non_offensive_series_rolling_mean.to_list()
    
    dates_joy_offensive_series = pd.Series(dates_joy_offensive_map, index=np.datetime_as_string(joy_in_offensive_dates))
    dates_joy_offensive_series_rolling = dates_joy_offensive_series.rolling(window=7)
    dates_joy_offensive_series_rolling_mean = dates_joy_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    joy_offensive_dates = dates_joy_offensive_series_rolling_mean.index.to_list()
    joy_offensive_dates = [np.datetime64(date) for date in joy_offensive_dates]
    num_joy_offensive = dates_joy_offensive_series_rolling_mean.to_list()
    
    dates_joy_non_offensive_series = pd.Series(dates_joy_non_offensive_map, index=np.datetime_as_string(joy_in_non_offensive_dates))
    dates_joy_non_offensive_series_rolling = dates_joy_non_offensive_series.rolling(window=7)
    dates_joy_non_offensive_series_rolling_mean = dates_joy_non_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    joy_non_offensive_dates = dates_joy_non_offensive_series_rolling_mean.index.to_list()
    joy_non_offensive_dates = [np.datetime64(date) for date in joy_non_offensive_dates]
    num_joy_non_offensive = dates_joy_non_offensive_series_rolling_mean.to_list()
    
    dates_optimism_offensive_series = pd.Series(dates_optimism_offensive_map, index=np.datetime_as_string(optimism_in_offensive_dates))
    dates_optimism_offensive_series_rolling = dates_optimism_offensive_series.rolling(window=7)
    dates_optimism_offensive_series_rolling_mean = dates_optimism_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    optimism_offensive_dates = dates_optimism_offensive_series_rolling_mean.index.to_list()
    optimism_offensive_dates = [np.datetime64(date) for date in optimism_offensive_dates]
    num_optimism_offensive = dates_optimism_offensive_series_rolling_mean.to_list()
    
    dates_optimism_non_offensive_series = pd.Series(dates_optimism_non_offensive_map, index=np.datetime_as_string(optimism_in_non_offensive_dates))
    dates_optimism_non_offensive_series_rolling = dates_optimism_non_offensive_series.rolling(window=7)
    dates_optimism_non_offensive_series_rolling_mean = dates_optimism_non_offensive_series_rolling.mean()/(num_offensive+num_non_offensive)*100
    optimism_non_offensive_dates = dates_optimism_non_offensive_series_rolling_mean.index.to_list()
    optimism_non_offensive_dates = [np.datetime64(date) for date in optimism_non_offensive_dates]
    num_optimism_non_offensive = dates_optimism_non_offensive_series_rolling_mean.to_list()
    
    # Plot Daily tweets, % of emotions wrt offensive tweets, % of emotions wrt non-offensive tweets,
    month_year_formatter = mdates.DateFormatter('%b %Y')
    labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Optimism']
    x = np.arange(len(labels))
    ncols = 2
    nrows = 1
    
    processing = save_as.split('_')[1]
    anger_line, disgust_line, fear_line, joy_line, optimism_line = None, None, None, None, None
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 2.7), layout="constrained")#12, 2.7
    # add an artist, in this case a nice label in the middle...
    for col in range(ncols):
        if col == 0:
            # Emotions of offensive tweets 
            axs[col].xaxis.set_major_formatter(month_year_formatter)
            for label in axs[col].get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right') 
            axs[col].plot(anger_offensive_dates, num_anger_offensive, label='Anger', linewidth=1)
            axs[col].plot(disgust_offensive_dates, num_disgust_offensive, label='Disgust', linewidth=1)
            axs[col].plot(fear_offensive_dates, num_fear_offensive, label='Fear', linewidth=1)
            axs[col].plot(joy_offensive_dates, num_joy_offensive, label='Joy', linewidth=1)
            axs[col].plot(optimism_offensive_dates, num_optimism_offensive, label='Optimism', linewidth=1)
            
            # plot vertical line
            for i, date in enumerate(emotions_in_offensive_tweets_vlines):
                axs[col].axvline(date, ls='--', color='#929591')
                if processing == "offenders":
                    axs[col].text(date, 1.2, events_lines[i], fontsize=8, va="center", ha="center")
                elif processing == "receivers":
                    axs[col].text(date, 0.15, events_lines[i], fontsize=8, va="center", ha="center")

            axs[col].set_ylabel('% of Tweets')
            axs[col].set_yscale('log')
            #axs[col].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*100:.2f}'))
            axs[col].yaxis.set_major_formatter(PercentFormatter())
            axs[col].set_title("A) Proportion of Offensive Tweets by Emotion", loc='left', fontsize=10, fontweight='medium', pad=15)
            #axs[col].legend()
            axs[col].grid(True)

        else:
            # Emotions of non-offensive tweets
            axs[col].xaxis.set_major_formatter(month_year_formatter)
            for label in axs[col].get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right') 
            anger_line = axs[col].plot(anger_non_offensive_dates, num_anger_non_offensive, label='Anger', linewidth=1)
            disgust_line = axs[col].plot(disgust_non_offensive_dates, num_disgust_non_offensive, label='Disgust', linewidth=1)
            fear_line = axs[col].plot(fear_non_offensive_dates, num_fear_non_offensive, label='Fear', linewidth=1)
            joy_line = axs[col].plot(joy_non_offensive_dates, num_joy_non_offensive, label='Joy', linewidth=1)
            optimism_line = axs[col].plot(optimism_non_offensive_dates, num_optimism_non_offensive, label='Optimism', linewidth=1)

            # plot vertical line
            for i, date in enumerate(emotions_in_non_offensive_tweets_vlines):
                axs[col].axvline(date, ls='--', color='#929591')
                if processing == "offenders":
                    if i == 1:
                        axs[col].text(emotions_in_non_offensive_tweets_vlines[i], 1.5, events_lines[i], fontsize=8, ha='right', va='center')
                    else:
                        axs[col].text(date, 1.5, events_lines[i], fontsize=8, ha='center', va='center')
                elif processing == "receivers":
                    if i == 1:
                        axs[col].text(emotions_in_non_offensive_tweets_vlines[i], 2.0, events_lines[i], fontsize=8, ha='right', va='center')
                    else:
                        axs[col].text(date, 2.0, events_lines[i], fontsize=8, ha='center', va='center')

            axs[col].set_ylabel('% of Tweets')
            axs[col].set_yscale('log')
            #axs[col].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
            axs[col].yaxis.set_major_formatter(PercentFormatter())
            axs[col].set_title("B) Proportion of Non-offensive Tweets by Emotion", loc='left', fontsize=10, fontweight='medium', pad=15)
            axs[col].grid(True)
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(default_colors, labels)]
    lines = [anger_line, disgust_line, fear_line, joy_line, optimism_line]
    fig.legend(labels=labels, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.1), handles=patches, handlelength=1.0)
    try:
        fig.savefig(f'figures_offenders_receivers/prop_emotions{save_as}.pdf', format='pdf', bbox_inches='tight')
    except FileNotFoundError:
        # this allows the script to keep going if run interactively and
        # the directory above doesn't exist
        pass
    
    # Plot bar chart of emotions in 2020, 2021, and 2022 all in one figure
    labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Optimism']
    x = np.arange(len(labels))
    width = 0.35
    ncols = 3
    nrows = 1
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 2.7), layout="constrained")#12, 3
    # add an artist, in this case a nice label in the middle...
    for col in range(3):        
        if col == 0:
            # Bar chart of emotions in 2020
            perc_anger_non_offensive = sum(num_anger_in_non_offensive_tweets_2020)
            perc_anger_offensive = sum(num_anger_in_offensive_tweets_2020)
            perc_disgust_non_offensive = sum(num_disgust_in_non_offensive_tweets_2020)
            perc_disgust_offensive = sum(num_disgust_in_offensive_tweets_2020)
            perc_fear_non_offensive = sum(num_fear_in_non_offensive_tweets_2020)
            perc_fear_offensive = sum(num_fear_in_offensive_tweets_2020)
            perc_joy_non_offensive = sum(num_joy_in_non_offensive_tweets_2020)
            perc_joy_offensive = sum(num_joy_in_offensive_tweets_2020)
            perc_optimism_non_offensive = sum(num_optimism_in_non_offensive_tweets_2020)
            perc_optimism_offensive = sum(num_optimism_in_offensive_tweets_2020)

            all_non_offensive = [
                round(perc_anger_non_offensive,2), 
                round(perc_disgust_non_offensive,2), 
                round(perc_fear_non_offensive,2), 
                round(perc_joy_non_offensive,2), 
                round(perc_optimism_non_offensive,2),
                                ]

            all_offensive = [
                round(perc_anger_offensive,2), 
                round(perc_disgust_offensive,2), 
                round(perc_fear_offensive,2), 
                round(perc_joy_offensive,2), 
                round(perc_optimism_offensive,2),
                                ]

            rects1 = axs[col].bar(x - width/2, all_non_offensive, width, label='Non-offensive')
            rects2 = axs[col].bar(x + width/2, all_offensive, width, label='Offensive')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            axs[col].set_ylabel('% of Tweets')
            axs[col].set_title('C) 2020 Tweets Emotional Distribution', loc='left', fontsize=10, fontweight='medium')
            axs[col].set_xticks(x, labels)
            for label in axs[col].get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
            #axs[row, col].legend()
            # axs[col].bar_label(rects1, padding=3)
            # axs[col].bar_label(rects2, padding=3)

        elif col == 1:
            # Bar chart of emotions in 2021
            perc_anger_non_offensive = sum(num_anger_in_non_offensive_tweets_2021)
            perc_anger_offensive = sum(num_anger_in_offensive_tweets_2021)
            perc_disgust_non_offensive = sum(num_disgust_in_non_offensive_tweets_2021)
            perc_disgust_offensive = sum(num_disgust_in_offensive_tweets_2021)
            perc_fear_non_offensive = sum(num_fear_in_non_offensive_tweets_2021)
            perc_fear_offensive = sum(num_fear_in_offensive_tweets_2021)
            perc_joy_non_offensive = sum(num_joy_in_non_offensive_tweets_2021)
            perc_joy_offensive = sum(num_joy_in_offensive_tweets_2021)
            perc_optimism_non_offensive = sum(num_optimism_in_non_offensive_tweets_2021)
            perc_optimism_offensive = sum(num_optimism_in_offensive_tweets_2021)

            all_non_offensive = [
                round(perc_anger_non_offensive,2), 
                round(perc_disgust_non_offensive,2), 
                round(perc_fear_non_offensive,2), 
                round(perc_joy_non_offensive,2), 
                round(perc_optimism_non_offensive,2),
                                ]

            all_offensive = [
                round(perc_anger_offensive,2), 
                round(perc_disgust_offensive,2), 
                round(perc_fear_offensive,2), 
                round(perc_joy_offensive,2), 
                round(perc_optimism_offensive,2),
                                ]

            rects1 = axs[col].bar(x - width/2, all_non_offensive, width, label='Non-offensive')
            rects2 = axs[col].bar(x + width/2, all_offensive, width, label='Offensive')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            axs[col].set_ylabel('% of Tweets')
            axs[col].set_title('D) 2021 Tweets Emotional Distribution', loc='left', fontsize=10, fontweight='medium')
            axs[col].set_xticks(x, labels)
            for label in axs[col].get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
            #axs[row, col].legend()
            # axs[col].bar_label(rects1, padding=3)
            # axs[col].bar_label(rects2, padding=3)

        elif col == 2:
            # Bar chart of emotions in 2022
            perc_anger_non_offensive = sum(num_anger_in_non_offensive_tweets_2022)
            perc_anger_offensive = sum(num_anger_in_offensive_tweets_2022)
            perc_disgust_non_offensive = sum(num_disgust_in_non_offensive_tweets_2022)
            perc_disgust_offensive = sum(num_disgust_in_offensive_tweets_2022)
            perc_fear_non_offensive = sum(num_fear_in_non_offensive_tweets_2022)
            perc_fear_offensive = sum(num_fear_in_offensive_tweets_2022)
            perc_joy_non_offensive = sum(num_joy_in_non_offensive_tweets_2022)
            perc_joy_offensive = sum(num_joy_in_offensive_tweets_2022)
            perc_optimism_non_offensive = sum(num_optimism_in_non_offensive_tweets_2022)
            perc_optimism_offensive = sum(num_optimism_in_offensive_tweets_2022)

            all_non_offensive = [
                round(perc_anger_non_offensive,2), 
                round(perc_disgust_non_offensive,2), 
                round(perc_fear_non_offensive,2), 
                round(perc_joy_non_offensive,2), 
                round(perc_optimism_non_offensive,2),
                                ]

            all_offensive = [
                round(perc_anger_offensive,2), 
                round(perc_disgust_offensive,2), 
                round(perc_fear_offensive,2), 
                round(perc_joy_offensive,2),
                round(perc_optimism_offensive,2)
                                ]

            rects1 = axs[col].bar(x - width/2, all_non_offensive, width, label='Non-offensive')
            rects2 = axs[col].bar(x + width/2, all_offensive, width, label='Offensive')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            axs[col].set_ylabel('% of Tweets')
            axs[col].set_title('E) 2022 Tweets Emotional Distribution', loc='left', fontsize=10, fontweight='medium')
            axs[col].set_xticks(x, labels)
            for label in axs[col].get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
            #axs[col].legend()
            # axs[col].bar_label(rects1, padding=3)
            # axs[col].bar_label(rects2, padding=3)
    
    labels = ['Non-offensive', 'offensive']
    fig.legend(labels=labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.52, 0.03))
    plt.savefig(f'figures_offenders_receivers/emotion_distributions_by_year{save_as}.pdf', format='pdf', bbox_inches='tight')