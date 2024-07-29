import csv
import os
import sys
import logging
import logging.handlers
from twarc import Twarc2, expansions


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


# Your bearer token here
TOKEN = "Your Twitter API bearer token"
client = Twarc2(bearer_token=TOKEN)


def get_tweet_ids(file_path):
    """
    Get tweet ids from file_path
    Arguments:
        datapath (String): the absolute path to the file_path. Each line contains tweet_id, author_id,
        date, screen name, username, sentiment, race, gender, emotions (11), offensive
    Returns:
        all_tweet_ids (List): A list of the tweet ids to check for compliant
    """
    
    logging.info("Gathering tweet ids ...")
    all_tweet_ids = []
    with open(file_path) as csv_file_handle:
        csv_reader = csv.reader(csv_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader):
            all_tweet_ids.append(line[0])
    logging.info("Tweet ids gathered ...")
    
    return all_tweet_ids

    
def get_non_compliant_tweet_ids(all_tweet_ids):
    """
    Determine tweets ids that are compliant and write the non compliant tweet ids to file
    Arguments:
        all_tweet_ids (List): A list of the tweet ids to check for compliance
    Returns:
        non_compliant_tweet_ids (List): A list of tweet ids that are not compliant with Twitter
    """
    
    logging.info("Checking Compliant tweets ...")
    compliant_tweet_ids = []

    # The tweet_lookup function will look up Tweets with the specified ids
    lookup = client.tweet_lookup(tweet_ids=all_tweet_ids)
    for i, page in enumerate(lookup):
        result = expansions.flatten(page)
        for tweet in result:
            compliant_tweet_ids.append(tweet['id'])
            
        logging.info(f"Checked compliance of {i + 1} tweets")
        
    # Here we get a difference betweetn the original
    non_compliant_tweet_ids = list(set(all_tweet_ids) - set(compliant_tweet_ids))
    logging.info(f"Compliance check completed.")
    
    return non_compliant_tweet_ids


def main():
    file_path = "../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/unique_tweets_2020_preprocessed_with_sentiment_race_gender_emotion_and_label_splits_combined_non_offensive_sampled.csv"
    all_tweet_ids = get_tweet_ids(file_path)
    non_compliant_tweet_ids = get_non_compliant_tweet_ids(all_tweet_ids)
    
    logging.info(f"Writing non compliant tweet ids to file ...")
    save_as_path = "../../../../zfs/socbd/eokpala/blm_research/data/"
    save_as = "non_compliant_2020_non_offensive_tweets.txt"
    with open(save_as_path + save_as, 'w') as new_file_handle:
        for tweet_id in non_compliant_tweet_ids:
            new_file_handle.write(f'{tweet_id}\n')
    logging.info(f"Non compliant tweet ids written to file.")

    
if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
    main()
