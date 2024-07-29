import json
import os
import random
import csv
import re
import emoji
from sklearn.model_selection import train_test_split
from pprint import pprint
from dateutil import parser
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import string

nltk.download('stopwords')
stopwords_english = stopwords.words('english')
stopwords_english_2 = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "the", "is"]
stopwords_english = set(stopwords_english + stopwords_english_2)


def get_next_token(file_location):
    """
    Return the next token used in the continuation of tweet collection from where collection stopped
    """
    
    with open(file_location) as file_handle:
        for line in file_handle:
            continue
    tweet = json.loads(line) # next token should be in the '__twarc' object
    url = tweet['__twarc']['url']
    url = url.split('&')
    last_item = url[-1]
    next_token = last_item.split('=')[-1]
    
    return next_token 


def count_tweets(file_name):
    """
    Counts the number of tweets in file_name.
    Arguments:
        file_name(String): The name of the file containing texts to count.
    Returns:
        None
    """
    
    num_tweets = 0
    with open(file_name) as file:
        for i, line in enumerate(file, 1):
            num_tweets += 1
    print(f"File: {file_name}, Number of tweets: {num_tweets} \n")


def count_offensive_tweets(file_name):
    
    num_offensive_tweets = 0
    num_non_offensive_tweets = 0
    with open(file_name) as input_file_handle:
        csv_reader = csv.reader(input_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader, 1):
            tweet_id = line[0]
            author_id = line[1]
            tweet = line[2]
            label = line[3]
            
            if int(label) == 0:
                num_non_offensive_tweets += 1
            elif int(label) == 1:
                num_offensive_tweets += 1
                
    print(f'Total number of tweets: {i}')
    print(f'Number of offensive tweets: {num_offensive_tweets}')
    print(f'Number of non offensive tweets: {num_non_offensive_tweets}')
    

def count_offensive_non_offensive_tweets(file_name):
    
    num_offensive_tweets = 0
    num_non_offensive_tweets = 0
    with open(file_name) as input_file_handle:
        csv_reader = csv.reader(input_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader, 1):
            label = line[-1]
            
            if int(label) == 0:
                num_non_offensive_tweets += 1
            elif int(label) == 1:
                num_offensive_tweets += 1
                
    print(f'Total number of tweets: {i}')
    print(f'Number of offensive tweets: {num_offensive_tweets}')
    print(f'Number of non offensive tweets: {num_non_offensive_tweets}')

    
def count_tweets_in_csv(file_name):
    '''Count the number of tweets in a csv file'''
    
    num_tweets = 0
    users = set()
    with open(file_name) as input_file_handle:
        csv_reader = csv.reader(input_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader, 1):
            author_id = line[1]
            users.add(author_id)
            num_tweets += 1
    
    print(file_name)
    print(f'Total number of tweets: {num_tweets}, unique users: {len(users)}')
    
    
def view_tweets(file_name, number_of_texts=1):
    """
    Prints number_of_texts in file_name to the screen. The output contains 
    the tweet id, author id, and tweet
    Arguments:
        file_name(String): The name of the file containing texts to print. Each line
        is a nested Dictionary of data
        number_of_texts(Int): The number of text to print, defaults to 5
    Returns:
        None
    """
    with open(file_name) as file:
        for i, line in enumerate(file, 1):
            tweet_object = json.loads(line)
            pprint(tweet_object)
            if i == number_of_texts:
                break


def view_parler_tweets(file_name, number_of_texts=1):
    
    with open(file_name) as input_file_handle:
        csv_reader = csv.reader(input_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader, 1):
            print(line)
            print()
            if i == number_of_texts:
                break
    
    
def remove_duplicate_tweets(file_path, save_as):
    """
    Removes duplicate tweets in a data set
    Args:
        file_path: String. The location of data set to remove duplicates from. 
        Each line is expected to be tweet json object
        save_as: String. Path to save the unique tweets 
    Returns:
        None
    """
    
    """
    seen = set()
    with open(save_as, 'a') as new_file:
        with open(file_path) as file:
            for i, line in enumerate(file, 1):
                tweet_object = json.loads(line)
                tweet = tweet_object['text'].strip()
                tweet = tweet.replace('\n', ' ')
                tweet = tweet.replace('\r', ' ')
                author_id = tweet_object['author_id']
                tweet_id = tweet_object['id']
                if tweet not in seen:
                    new_file.write(tweet_id + '\t' + author_id + '\t' + tweet + '\n')
                    seen.add(tweet)
    """
    with open(save_as, 'a') as new_file:
        with open(file_path) as current_file:
            lines = current_file.readlines()
            for i in range(0, len(lines), 2):
                new_file.write(lines[i].strip() + '\t' + lines[i+1].strip() + '\n')
            

def combine_yearly_datasets(parent_folder, dataset_names, save_as):
    '''
    Combine the yearly (2020, 2021, 2022) datasets into one dataset
    Arguments:
        parent_folder (String): The folder containing individual yearly data to be combined.
        dataset_names (List): A list of type string containing the dataset names
        save_as (String): The name of the .jsonl file to contain the combined data
    Returns:
        None
    '''
    
    with open(parent_folder + '/' + save_as, 'a') as new_file_handle:
        for dataset in dataset_names:
            with open(parent_folder + '/' + dataset) as dataset_file_handle:
                for line in dataset_file_handle:
                    new_file_handle.write(f"{line}")

                    
def combine_non_compliant_tweet_ids(offensive_non_compliant_file, non_offensive_non_compliant_file, save_as):
    '''
    Combine the offensive and non-offensive non-compliant tweet ids of each year so that change point detection can be performed. 
    Arguments:
        offensive_non_compliant_file (String): The path to the offensive non-compliant tweet ids file in txt format
        non_offensive_non_compliant_file (String): The path to the non-offensive non-compliant tweets file in txt format
        save_as (String): The path to store the combined tweets
    Returns:
        None
    '''
    
    with open(save_as, mode='w') as file_handle:
        with open(offensive_non_compliant_file) as offensive_file_handle:
            for row in offensive_file_handle:
                file_handle.write(f'{row}')
                
        with open(non_offensive_non_compliant_file) as non_offensive_file_handle:
            for row in non_offensive_file_handle:
                file_handle.write(f'{row}')
                    

def extract_tweets(file_path, save_as):
    """
    Extract tweetID, authorID, and tweet text from file_path
    Arguments:
        file_path (String): Absolute path containing the data set for extraction. 
        dataset_name (String): The name of the dataset. Each line is expected to be a tweet json object.
        save_as (String): Absolute path to save the extracted details. File name in this is expected to be .txt
    Returns:
        None
    """
    
    # Remove duplicates
    seen = set()
    with open(save_as, 'a') as new_file_handle:
        with open(file_path) as file_handle:
            for line in file_handle:
                tweet_object = json.loads(line)
                tweet = tweet_object['text'].strip().strip('\n')
                tweet = tweet.replace('\n', ' ')
                tweet = tweet.replace('\r', ' ')
                author_id = tweet_object['author_id']
                tweet_id = tweet_object['id']
                if tweet not in seen:
                    new_file_handle.write(tweet_id + '\t' + author_id + '\t' + tweet + '\n')
                    seen.add(tweet)


def sample_tweets(file, save_sampled_as, k=50000):
    """
    Randomly sample k tweets from file
    Arguments:
        file (String): Absolute path (path + file name) to file to sample tweets from. 
        Each line is tab-separated and of the form - tweetID authorID tweet. 
        save_sampled_as (String): Absolute path (path + file name) to where to save the sampled tweets and the name of the file
        k (Int): The number to sample. Defaults to 50,000
    Returns:
        None
    """
    
    with open(file) as combined_file:
        all_tweets = combined_file.readlines()
    
    seed = 5
    random.seed(seed)
    sampled_tweets = random.sample(all_tweets, k=k)
    # save sampled tweets 
    with open(save_sampled_as, 'a') as sample_file:
        for tweet in sampled_tweets:
            sample_file.write(tweet)
            
            
def split_dataset(data_path, save_train_as, save_test_as, train_size=0.90):
    """
    Splits a dataset into train and test set. This function assumes there is
    a file containing text and label in each line separated by tab
    Args:
        data_path (String): path where the full dataset test set is contained
        save_train_as (String): absolute path to the training set
        save_test_as (String): absolute path to the test set
        train_size(Float): The percentage to divide the training set by, the test
        set will be 100 - train_split * 100. Defaults to 0.9 (90%). 
    Returns:
        None
    """

    data, labels = [], []
    seed = 42

    with open(data_path) as file_handler:
        for i, line in enumerate(file_handler):
            line_array = line.strip().split("\t")
            tweet_id = line_array[0]
            tweet = line_array[1]
            label = line_array[2]
            data.append((tweet, tweet_id))
            labels.append(label)

    # Create train and test set split
    train_data, test_data, train_label, test_label = train_test_split(
            data, labels, train_size=train_size, random_state=seed
        )
    
    # Write train set
    with open(save_train_as, "a+") as file_handler:
        for i, tweet in enumerate(train_data):
            tweet_text, tweet_id = tweet
            label = train_label[i]
            file_handler.write(tweet_id + '\t' + tweet_text + '\t' + label + '\n')

    # Write test set
    with open(save_test_as, "a+") as file_handler:
        for i, tweet in enumerate(test_data):
            tweet_text, tweet_id = tweet
            label = test_label[i]
            file_handler.write(tweet_id + '\t' + tweet_text + '\t' + label + '\n')


def split_offensive_dataset(data_path, save_train_as, save_test_as, train_size=0.90):
    """
    Splits a dataset into train and test set. This function assumes the file in
    data path is comma separated. 
    Args:
        data_path (String): path where the full dataset test set is contained
        save_train_as (String): absolute path to the training set
        save_test_as (String): absolute path to the test set
        train_size(Float): The percentage to divide the training set by, the test
        set will be 100 - train_split * 100. Defaults to 0.9 (90%). 
    Returns:
        None
    """

    data, labels = [], []
    seed = 23
    
    with open(data_path) as input_file_handle:
        csv_reader = csv.reader(input_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader):
            tweet_id = line[0]
            author_id = line[1]
            tweet = line[2]
            label = line[3]
            data.append((tweet_id, author_id, tweet))
            labels.append(label)

    # Create train and test set split
    train_data, test_data, train_label, test_label = train_test_split(
            data, labels, train_size=train_size, random_state=seed
        )
    
    # Write train set
    with open(save_train_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, tweet in enumerate(train_data):
            tweet_id, author_id, tweet_text = tweet
            label = train_label[i]
            row = [tweet_id, author_id, tweet_text, label]
            csv_handler.writerow(row)

    # Write test set
    with open(save_test_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, tweet in enumerate(test_data):
            tweet_id, author_id, tweet_text = tweet
            label = test_label[i]
            row = [tweet_id, author_id, tweet_text, label]
            csv_handler.writerow(row)
            
            
def handle_sentiment_dataset(file_path, save_as):
    """
    Handles the Sentiment140 dataset by extracting needed columns from the .csv file, 
    preprocessing, and converting labels to the state needed for training. 
    The sentniment140 dataset can be found at this link: http://help.sentiment140.com/for-students
    Args:
        file_path(String): the path to the data set. file_path is expected to be a .csv file
        save_as(String): the path to save the extracted tweets and lables in
    Returns:
        None
    """
    
    with open(save_as, 'a+') as to_text_file_handle:
        with open(file_path, encoding='latin-1') as csv_file_handle:
            csv_reader = csv.reader(csv_file_handle, delimiter=',')
            for i, line in enumerate(csv_reader):
                sentiment = line[0]
                tweet_id = line[1]
                tweet = line[5]
                tweet = tweet.replace("\n", " ")
                tweet = tweet.replace("\r", " ")
                tweet = tweet.replace("\t", " ")
                # 0 is negative sentiment - code it as 0 and code positive sentiment (4) as 1
                if sentiment == "0":
                    label = "0"
                elif sentiment == "4":
                    label = "1"

                tweet = preprocess_sentiment_tweet(tweet)
                to_text_file_handle.write(tweet_id + '\t' + tweet + '\t' + label + '\n')
                

def handle_offensive_blm_dataset(file_path, save_as):
    """
    Handles the labeled blm dataset. 
    Args:
        file_path(String): the path to the data set. file_path is expected to be a .csv file
        save_as(String): the path to save the extracted tweets and lables in
    Returns:
        None
    """
    
    seen = set()
    with open(save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open(file_path) as input_file_handle:
            csv_reader = csv.reader(input_file_handle, delimiter=',')
            for i, line in enumerate(csv_reader):
                tweet_id = line[0]
                author_id = line[1]
                tweet = line[2]
                label = line[3]
                tweet = tweet.replace("\n", " ")
                tweet = tweet.replace("\r", " ")
                tweet = tweet.replace("\t", " ")
                tweet = preprocess_tweet(tweet)
                row = [tweet_id, author_id, tweet, label]
                
                if tweet not in seen:
                    seen.add(tweet)
                    csv_handler.writerow(row)


def handle_parler_dataset(file_path, save_as):
    """
    Handles the labeled blm dataset. 
    Args:
        file_path(String): the path to the data set. file_path is expected to be a .csv file
        save_as(String): the path to save the extracted tweets and lables in
    Returns:
        None
    """
    
    seen = set()
    with open(save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open(file_path) as input_file_handle:
            csv_reader = csv.reader(input_file_handle, delimiter=',')
            for i, line in enumerate(csv_reader):
                comment_id = line[0]
                author_id = line[1]
                date = line[2]
                comment = line[3]
                comment = comment.replace("\n", " ")
                comment = comment.replace("\r", " ")
                comment = comment.replace("\t", " ")
                comment = preprocess_tweet(comment)
                
                if comment != None:
                    row = [comment_id, author_id, date, comment]
                    if comment not in seen:
                        seen.add(comment)
                        csv_handler.writerow(row)


def extract_parler_comments_by_year(file_path, save_as, year='2020'):
    
    with open(save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open(file_path) as input_file_handle:
            csv_reader = csv.reader(input_file_handle, delimiter=',')
            for i, line in enumerate(csv_reader):
                date = line[2]
                date_list = date.split('-')
                curr_year = date_list[0]
                month = date_list[1]
                months_set = {'05', '06', '07', '08', '09', '10', '11', '12'}
                if curr_year == year:
                    if month in months_set:
                        csv_handler.writerow(line)


def preprocess_sentiment_tweet(tweet):
    '''
    Proprocess tweets by lower casing, normalize by converting
    user mentions to @USER, url to HTTPURL, and number to NUMBER. 
    Convert emoji to text string and remove duplicate tweets.
    Download the dataset here: http://help.sentiment140.com/for-students
    Arguments:
        tweet(String): A sentiment tweet to preprocess
    Return:
        None
    '''
    
    tweet = tweet.strip()
    tweet = tweet.strip('"')
    tweet = tweet.lower()

    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^rt[\s]+', '', tweet)
    # replace hyperlinks with URL
    tweet = re.sub(r'(https?:\/\/[a-zA-Z0-9]+\.[^\s]{2,})', 'HTTPURL', tweet)
    # remove hashtags - only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # replace emojis with emoji text
    tweet = emoji.demojize(tweet, delimiters=("", ""))
    # remove "..." elipses
    tweet = re.sub(r'\.{2,}', '', tweet)
    # replace numbers with NUMBER
    tweet = re.sub(r'^\d+$', 'NUMBER', tweet)
    # replace handles with @USER
    tweet = re.sub(r'@\w+', '@USER', tweet)
    return tweet


def preprocess_tweet(tweet):
    '''
    Proprocess tweets by lower casing, normalize by converting
    user mentions to @USER, url to URL, and number to NUMBER. 
    Convert emoji to text string.
    Arguments:
        tweet(String): A tweet to preprocess
    Return:
        None
    '''
    
    tweet = tweet.strip()
    tweet = tweet.strip('"')
    tweet = tweet.lower()

    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^rt[\s]+', '', tweet)
    # replace hyperlinks with URL
    tweet = re.sub(r'(https?:\/\/[a-zA-Z0-9]+\.[^\s]{2,})', 'URL', tweet)
    # remove hashtags - only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # remove "..." elipses
    tweet = re.sub(r'\.{2,}', '', tweet)
    # replace numbers with NUMBER
    tweet = re.sub(r'\d+', 'NUMBER', tweet)
    # replace handles with @USER
    tweet = re.sub(r'@\w+', '@USER', tweet)
    
    # tokenize tweets
    tokenizer = TweetTokenizer(reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    clean_tweet = []
    
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            clean_tweet.append(word)
    
    if len(clean_tweet) > 3:
        tweet = " ".join(clean_tweet)
        # Replace emojis with their text equivalent 
        tweet = emoji.demojize(tweet)
        return tweet

    return None


def extract_potential_offensive_tweets(file_path, save_as, threshold=0.7):
    """
    Extract potential offensive tweets as tweets having a perspective score of >= threshold
    Arguments:
        file_path (String): Absolute path containing the data set for extraction. 
        save_as (String): Absolute path to save the extracted details. File name in this is expected to be .txt
        threshold(Float): The toxicity score to consider as potentially offensive
    Returns:
        None
    """
    
    with open(save_as, 'a') as new_file_handle:
        with open(file_path) as file_handle:
            for line in file_handle:
                line_array = line.split('\t')
                tweet_id = line_array[0].strip()
                author_id = line_array[1].strip()
                tweet = line_array[2].strip()
                score = line_array[3].strip()
                
                if float(score) >= threshold:
                    new_file_handle.write(tweet_id + '\t' + author_id + '\t' + tweet + '\n')
                    

def extract_tweets_with_negative_sentiment(file_path, save_as):
    """
    Extract tweets having negative sentiment from tweets with high toxicity score
    Arguments:
        file_path (String): Absolute path containing the data set for extraction. 
        save_as (String): Absolute path to save the extracted details. File name in this is expected to be .txt
    Returns:
        None
    """
    
    with open(save_as, 'a') as new_file_handle:
        with open(file_path) as file_handle:
            for line in file_handle:
                line_array = line.split('\t')
                tweet_id = line_array[0].strip()
                author_id = line_array[1].strip()
                tweet = line_array[2].strip()
                sentiment = line_array[3].strip()
                
                if int(sentiment) == 0: # 0 means negative
                    new_file_handle.write(tweet_id + '\t' + author_id + '\t' + tweet + '\n')


def write_tweets_with_negative_sentiment_to_csv(file_path, save_as):
    '''
    Write data in .txt to .csv to make labeling convenient
    Arguments:
        file_path (String): Absolute path containing the data set in .txt. 
        save_as (String): Absolute path to save the extracted details.
    Returns:
        None
    '''
    with open(save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open(file_path) as file_handle:
            for line in file_handle:
                line_array = line.split('\t')
                tweet_id = line_array[0].strip()
                author_id = line_array[1].strip()
                tweet = line_array[2].strip()
                row = [tweet_id, author_id, tweet]
                csv_handler.writerow(row)


def retrieve_unique_tweets_from_combined_dataset(file_path, 
                                                 save_combined_as,
                                                 save_preprocessed_combined_as,
                                                 save_2020_as,
                                                 save_preprocessed_2020_as,
                                                 save_2021_as,
                                                 save_preprocessed_2021_as,
                                                 save_2022_as,
                                                 save_preprocessed_2022_as):
    '''
    Get all the unique tweets in combined dataset file, extract and save tweet_id, author_id, tweet, date, name, and username
    Args:
        file_path (String): location of combined dataset file 
        save_combined_as (String): location and name to save all the unique tweets
        save_2020_as (String): location and name to save the unique tweets from 2020
        save_2021_as (String): location and name to save the unique tweets from 2021
        save_2021_as (String): location and name to save the unique tweets from 2022
    Returns:
        None
    '''
    
    seen = set()
    with (
        open(save_combined_as, mode='w') as csv_file_handle_combined,
        open(save_preprocessed_combined_as, mode='w') as csv_file_handle_preprocessed_combined,
        open(save_2020_as, mode='w') as csv_file_handle_2020,
        open(save_preprocessed_2020_as, mode='w') as csv_file_handle_preprocessed_2020,
        open(save_2021_as, mode='w') as csv_file_handle_2021,
        open(save_preprocessed_2021_as, mode='w') as csv_file_handle_preprocessed_2021,
        open(save_2022_as, mode='w') as csv_file_handle_2022,
        open(save_preprocessed_2022_as, mode='w') as csv_file_handle_preprocessed_2022
    ):
        csv_combined_handler = csv.writer(csv_file_handle_combined, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_preprocessed_combined_handler = csv.writer(csv_file_handle_preprocessed_combined, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_2020_handler = csv.writer(csv_file_handle_2020, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_preprocessed_2020_handler = csv.writer(csv_file_handle_preprocessed_2020, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_2021_handler = csv.writer(csv_file_handle_2021, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_preprocessed_2021_handler = csv.writer(csv_file_handle_preprocessed_2021, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_2022_handler = csv.writer(csv_file_handle_2022, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_preprocessed_2022_handler = csv.writer(csv_file_handle_preprocessed_2022, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        with open(file_path) as file_handle:
            for i, line in enumerate(file_handle, 1):
                tweet_object = json.loads(line)
                tweet = tweet_object['text'].strip()
                tweet = tweet.replace("\n", " ")
                tweet = tweet.replace("\r", " ")
                tweet = tweet.replace("\t", " ")
                author_id = tweet_object['author_id']
                tweet_id = tweet_object['id']
                date = parser.parse(tweet_object['created_at']).date().strftime('%Y-%m-%d')
                name = tweet_object['author']['name'].strip().replace("\x00", "").replace("\xa0\xa0", "").replace("\xa0", "")  #Remove null byte characters from string
                username = tweet_object['author']['username']
                
                if tweet not in seen:
                    seen.add(tweet)
                    preprocessed_tweet = preprocess_tweet(tweet)
                    if preprocessed_tweet != None: # Only save tweets that have more than three words
                        if name != "": 
                            row_with_preprocessed_tweet = [tweet_id, author_id, preprocessed_tweet, date, name, username]
                            row_without_preprocessed_tweet = [tweet_id, author_id, tweet, date, name, username]
                            csv_combined_handler.writerow(row_without_preprocessed_tweet)
                            csv_preprocessed_combined_handler.writerow(row_with_preprocessed_tweet)
                            year = date.split('-')[0]

                            if year == '2020':
                                csv_2020_handler.writerow(row_without_preprocessed_tweet)
                                csv_preprocessed_2020_handler.writerow(row_with_preprocessed_tweet)
                            elif year == '2021':
                                csv_2021_handler.writerow(row_without_preprocessed_tweet)
                                csv_preprocessed_2021_handler.writerow(row_with_preprocessed_tweet)
                            elif year == '2022':
                                csv_2022_handler.writerow(row_without_preprocessed_tweet)
                                csv_preprocessed_2022_handler.writerow(row_with_preprocessed_tweet)                   


def handle_emotion_dataset(file_path, save_as):
    '''Data located at: https://competitions.codalab.org/competitions/17751'''
    with open(save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open(file_path) as file_handle:
            for i, line in enumerate(file_handle):
                if i == 0:
                    continue
                line_array = line.split('\t')
                Id = line_array[0].strip()
                tweet = line_array[1].strip().replace("\x00", "")
                tweet = tweet.strip().replace("\xa0", "")
                tweet = tweet.replace("\n", " ")
                tweet = tweet.replace("\r", " ")
                tweet = tweet.replace("\t", " ")
                tweet = preprocess_tweet(tweet)
                anger = line_array[2].strip()
                anticipation = line_array[3].strip()
                disgust = line_array[4].strip()
                fear = line_array[5].strip()
                joy = line_array[6].strip()
                love = line_array[7].strip()
                optimism = line_array[8].strip()
                pessimism = line_array[9].strip()
                sadness = line_array[10].strip()
                surprise = line_array[11].strip()
                trust = line_array[12].strip().replace("\n", "")
                row = [Id, tweet, anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust]
                csv_handler.writerow(row)
    

def divide_datasets_for_faster_race_inference(path, file_path, num_tweets, save_as):
    '''
    The 2020 dataset is too large to perform race inference on within the 3 days resource
    limit we are bound to.
    So we split the remaining uninfered into multiple files to allow for faster inference.
    Args:
        path (String): base path where all data is stored
        file_path (String): the file containing the 2020 dataset with infered sentiment
        num_tweets (Int): The number of tweets to write to each new file 
        save_as (String): The name (path + name) where the splits will be stored
    Returns:
        None
    '''
    
    reached_num_tweets = 0
    num_tweets_written = 0
    unique_id = 0
    
    save_as = save_as + "_split_"
    with open(file_path) as csv_file_handle1: # The huge 2020 dataset is unique_tweets_2020_preprocessed_with_sentiment.csv
        csv_reader1 = csv.reader(csv_file_handle1, delimiter=',')
        for i, row in enumerate(csv_reader1):            
            if reached_num_tweets == 0:
                with open(path + save_as + str(unique_id) + '.csv', mode='a') as csv_file_handle2:
                    csv_handler2 = csv.writer(csv_file_handle2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_handler2.writerow(row)
                    num_tweets_written += 1
                    if num_tweets_written == num_tweets:
                        # Set flag that we have reached the desired number of tweets we need in a new file
                        reached_num_tweets = 1
                        # Reset number of tweets written to zero so the number of tweets to be written in the next file can be counted correctly
                        num_tweets_written = 0
                        # Increment unique id flag that separates each file
                        unique_id += 1
            else:
                with open(path + save_as + str(unique_id) + '.csv', mode='a') as csv_file_handle3:
                    csv_handler3 = csv.writer(csv_file_handle3, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_handler3.writerow(row)
                    num_tweets_written += 1
                    if num_tweets_written == num_tweets:
                        reached_num_tweets = 0
                        num_tweets_written = 0
                        unique_id += 1
        
        
def combine_split_2020_datasets(data_paths, save_as):
    '''
    Combine all the split 2020 datasets into a single file
    '''
    
    with open(save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for file_path in data_paths:
            with open(file_path) as csv_file_path_handle:
                csv_reader = csv.reader(csv_file_path_handle, delimiter=',')
                for i, row in enumerate(csv_reader):
                    csv_handler.writerow(row)

        
def sample_from_non_offensive_tweets(datasets, k=2000000):
    """
    The 2020 and some 2021 non-offensive tweets are large which requires a lot of memory by BERTopic.
    So sample 2M tweets from the 2020 and some 2021 non-offensive tweets datasets.
    """
    
    seed = 5
    random.seed(seed)
    
    for dataset in datasets:
        non_offensive_tweets = []
        with open(dataset) as csv_file_path_handle:
            csv_reader = csv.reader(csv_file_path_handle, delimiter=',')
            for i, row in enumerate(csv_reader):
                non_offensive_tweets.append(row)
    
        sampled_tweets = random.sample(non_offensive_tweets, k=k)
        # save sampled tweets 
        path_tokens = dataset.split('/')
        file_name = path_tokens[-1].split('.')[0]
        save_as = "/".join(path_tokens[:-1]) + '/' + file_name + '_sampled' + '.csv'
        with open(save_as, 'w') as file_handle:
            sampled_handler = csv.writer(file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for data in sampled_tweets:
                sampled_handler.writerow(data)
                
                
def count_non_compliant_tweets(file_path):
    '''Count the number of non compliant tweets in file path. Each line in file path is a tweet id'''
    
    with open(file_path) as file_handle:
        all_tweet_ids = file_handle.read().splitlines()
        
    print(f'Number of non compliant tweets: {len(all_tweet_ids)}')
    

def extract_compliant_tweets(non_compliant_ids_file_path, tweets_file_path, save_as):
    '''
    The offensive and non-offensive tweets copora for each year contain tweets that are non-compliant (deleted tweets or suspended accounts). 
    Extract the tweets that are compliant in the offensive and non-offensive tweets.
    Arguments:
        non_compliant_ids_file_path (String): Absolute path to the file containing the tweet ids of non-compliant tweets
        tweets_file_path (String): Absolute path to the file containing the offensive or non-offensive tweets each line is of the
        form tweet_id,author_id,tweet
        save_as (String): Absolute path of the file that will contain the compliant tweets
    Returns:
        None
    '''
    
    # Read the tweet ids of the non-compliant tweets
    with open(non_compliant_ids_file_path) as non_compliant_file_handle:
        all_tweet_ids = non_compliant_file_handle.read().splitlines()
        all_tweet_ids = set(all_tweet_ids)
        
    # Open a new file to save only the compliant tweets 
    with open(save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        # Read the offensive or non-offensive tweets and extract only compliant tweets
        with open(tweets_file_path) as csv_file_path_handle:
            csv_reader = csv.reader(csv_file_path_handle, delimiter=',')
            for i, row in enumerate(csv_reader):
                tweet_id = row[0]
                if tweet_id not in all_tweet_ids:
                    csv_handler.writerow(row)

                    
def extract_compliant_tweets2(compliant_offensive_file, compliant_non_offensive_file, all_tweets, save_as):
    '''
    The compliant offensive and non-offensive tweets contain tweets that are compliant but each line in the files are not 
    in the desired format to work with the change point detection method. 
    Extract the tweets that are compliant in the offensive and non-offensive tweets from all tweets. 
    The format of all_tweets; each line is of the form 
    tweetid 0, authorid 1, tweet 2, date 3, name 4, username 5, sentiment 6, race 7, gender 8, anger 9, anticipation 10,
    disgust 11, fear 12, joy 13, love 14, optimism 15, pessimism 16, sadness 17, surprise 18, trust 19, offensive 20
    Arguments:
        compliant_offensive_file (String): Absolute path to the file containing the tweet ids of non-compliant tweets
        compliant_non_offensive_file (String): Absolute path to the file containing the offensive or non-offensive tweets. Each line is of the form tweet_id,author_id,tweet
        all_tweets (String): Absolute path to the file containing both offensive and non-offensive tweets in the right format. 
        save_as (String): Absolute path of the file that will contain the compliant tweets
    Returns:
        None
    '''
        
    # Open a new file to save only the compliant tweets 
    csv_file_handle = open(save_as, mode='w')
    csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    offensive_tweet_ids = set()
    # Read the offensive tweets
    with open(compliant_offensive_file) as csv_offensive_file_path_handle:
        csv_offensive_reader = csv.reader(csv_offensive_file_path_handle, delimiter=',')
        for i, row in enumerate(csv_offensive_reader):
            tweet_id = row[0]
            offensive_tweet_ids.add(tweet_id)
            
    non_offensive_tweet_ids = set()
    # Read the non-offensive tweets
    with open(compliant_non_offensive_file) as csv_non_offensive_file_path_handle:
        csv_non_offensive_reader = csv.reader(csv_non_offensive_file_path_handle, delimiter=',')
        for i, row in enumerate(csv_non_offensive_reader):
            tweet_id = row[0]
            non_offensive_tweet_ids.add(tweet_id)
            
            
    # Go through all tweets 
    with open(all_tweets) as csv_all_tweets_file_path_handle:
        csv_all_tweets_reader = csv.reader(csv_all_tweets_file_path_handle, delimiter=',')
        for i, row in enumerate(csv_all_tweets_reader):
            tweet_id = row[0]
            if tweet_id in offensive_tweet_ids or tweet_id in non_offensive_tweet_ids:
                # Write to new file
                csv_handler.writerow(row)

    csv_file_handle.close()
        

def combine_compliant_tweets(compliant_offensive_file, compliant_non_offensive_file, all_tweets, save_as):    
        
    # Open a new file to save only the combined tweets 
    csv_file_handle = open(save_as, mode='w')
    csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                
    offensive_tweet_ids = set()
    # Read the offensive tweets
    with open(compliant_offensive_file) as csv_offensive_file_path_handle:
        csv_offensive_reader = csv.reader(csv_offensive_file_path_handle, delimiter=',')
        for i, row in enumerate(csv_offensive_reader):
            tweet_id = row[0]
            offensive_tweet_ids.add(tweet_id)
            
    non_offensive_tweet_ids = set()
    # Read the non-offensive tweets
    with open(compliant_non_offensive_file) as csv_non_offensive_file_path_handle:
        csv_non_offensive_reader = csv.reader(csv_non_offensive_file_path_handle, delimiter=',')
        for i, row in enumerate(csv_non_offensive_reader):
            tweet_id = row[0]
            non_offensive_tweet_ids.add(tweet_id)
            
    # Go through all tweets 
    with open(all_tweets) as csv_all_tweets_file_path_handle:
        csv_all_tweets_reader = csv.reader(csv_all_tweets_file_path_handle, delimiter=',')
        for i, row in enumerate(csv_all_tweets_reader):
            tweet_id = row[0]
            if tweet_id in offensive_tweet_ids or tweet_id in non_offensive_tweet_ids:
                # Write the full details to new file
                csv_handler.writerow(row)

    csv_file_handle.close()
    
    
def combine_compliant_and_non_compliant_tweets(compliant_offensive_file, non_compliant_non_offensive_file, all_tweets, save_as):    
        
    # Open a new file to save only the combined tweets 
    csv_file_handle = open(save_as, mode='w')
    csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                
    offensive_tweet_ids = set()
    # Read the offensive tweets
    with open(compliant_offensive_file) as csv_offensive_file_path_handle:
        csv_offensive_reader = csv.reader(csv_offensive_file_path_handle, delimiter=',')
        for i, row in enumerate(csv_offensive_reader):
            tweet_id = row[0]
            offensive_tweet_ids.add(tweet_id)
            
    non_offensive_tweet_ids = set()
    # Read the non-offensive tweets
    with open(non_compliant_non_offensive_file) as csv_non_offensive_file_path_handle:
        csv_non_offensive_reader = csv.reader(csv_non_offensive_file_path_handle, delimiter=',')
        for i, row in enumerate(csv_non_offensive_reader):
            tweet_id = row[0]
            non_offensive_tweet_ids.add(tweet_id)
            
    # Go through all tweets 
    with open(all_tweets) as csv_all_tweets_file_path_handle:
        csv_all_tweets_reader = csv.reader(csv_all_tweets_file_path_handle, delimiter=',')
        for i, row in enumerate(csv_all_tweets_reader):
            tweet_id = row[0]
            if tweet_id in offensive_tweet_ids or tweet_id in non_offensive_tweet_ids:
                # Write the full details to new file
                csv_handler.writerow(row)

    csv_file_handle.close()