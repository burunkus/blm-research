import json
import datetime
import sys
import csv
import os
import logging
import logging.handlers

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


def create_edge_list(original_tweets_file, extract_from, save_as):
    
    logging.info(f"Getting the compliant users ...")
    # Get the user ids
    compliant_user_details = set()
    with open(extract_from) as csv_file_path_handle:
        csv_reader = csv.reader(csv_file_path_handle, delimiter=',')
        for i, row in enumerate(csv_reader):
            compliant_user_details.add((row[0], row[1])) # (tweetid, userid)
    logging.info(f"Users retrieved.")
    
    logging.info(f"Going through each file in the original dataset ...")
    edge_list = {}
    # Lookup the original tweet object of users in user_ids and extract who they replied to or mentioned
    with open(original_tweets_file) as file_handle:
        for i, tweet in enumerate(file_handle):
            tweet_object = json.loads(tweet)
            user_id = tweet_object['author_id']
            tweet_id = tweet_object['id']
            
            user_detail = (tweet_id, user_id)
            
            if user_detail in compliant_user_details:
                tweet = tweet_object['text'].strip().strip('\n')
                tweet = tweet.replace('\n', ' ')
                tweet = tweet.replace('\r', ' ')
                
                replied_to = None
                
                # Get who the user replied to if any
                if 'in_reply_to_user_id' in tweet_object:
                    replied_to = tweet_object['in_reply_to_user_id']
                
                # Ensure their was a reply and avoid self loop
                if replied_to and user_detail[1] != replied_to:
                    user_id = user_detail[1]
                    # If this edge (u, v) is already existing increase the number of offensive tweets from u to v
                    if (user_id, replied_to) in edge_list:
                        edge_list[(user_id, replied_to)] += 1
                    else:
                        # Add edge (u, v) to the dictionary and set the number of offensive tweets from u to v to 1
                        edge_list[(user_id, replied_to)] = 1
            #logging.info(f"Tweets processed: {i + 1}")
            
    logging.info(f"Writing edge list to a file ...")        
    # Write the adjacency list to file 
    with open(save_as + '.edgelist', 'w') as text_file_handle:
        for (u, v), weight in edge_list.items():
            text_file_handle.write(f'{u} {v} {weight}\n')
    
    logging.info(f"Adjacency list written to a file.")
    
    
def main():
    # We need to collect the user id of those a user replied to
    original_tweets_files = [
        "../../../../../zfs/socbd/eokpala/blm_research/data/tweets_2020.jsonl",
        "../../../../../zfs/socbd/eokpala/blm_research/data/tweets_2021.jsonl",
        "../../../../../zfs/socbd/eokpala/blm_research/data/tweets_2022.jsonl"
    ]
    extract_from = [
        "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_offensive_compliant.csv",
        "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2021_preprocessed_with_sentiment_race_gender_emotion_and_label_offensive_compliant.csv",
        "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2022_preprocessed_with_sentiment_race_gender_emotion_and_label_offensive_compliant.csv"
    ]
    save_as = [
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2020_reply_tweets_edge_list",
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2021_reply_tweets_edge_list",
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2022_reply_tweets_edge_list"
    ]
    for i, year_file_path in enumerate(original_tweets_files):
        create_edge_list(year_file_path, extract_from[i], save_as[i])


if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
    main()