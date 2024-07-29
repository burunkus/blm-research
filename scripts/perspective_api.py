from googleapiclient import discovery
import json
import time

# See https://developers.perspectiveapi.com/s/docs-get-started?language=en_US 
# For how to obtain Perspective API key
API_KEY = 'Your perspective API key' 
client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

def get_toxicity_score(sampled_data_path, save_as):
    '''
    Use the Perspective API to retrieve the toxicity score of each tweet in the sampled tweets
    Arguments:
        sampled_data_path (String): The absolute path to the sampled data
        save_as (String): The path (absolute path + file name) to save the tweet and the score
    Returns:
        None
    '''
    
    with open(save_as, 'a') as new_file_handle:
        with open(sampled_data_path) as file_handle:
            for line in file_handle:
                line = line.split('\t')
                tweet_id = line[0].strip()
                author_id = line[1].strip()
                tweet = line[2].strip()
    
                analyze_request = {
                  'comment': { 'text': tweet },
                  'languages': ['en'],
                  'requestedAttributes': {'TOXICITY': {}},
                  'doNotStore': True
                }
                response = client.comments().analyze(body=analyze_request).execute()
                score = response['attributeScores']['TOXICITY']['summaryScore']['value']
                new_file_handle.write(tweet_id + '\t' + author_id + '\t' + tweet + '\t' + str(score) + '\n')
                time.sleep(2) # Sleep for 2 seconds before processing next request because of the API's request limit


if __name__ == "__main__":
    sampled_data_path = '../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets.txt'
    save_as = '../../../../zfs/socbd/eokpala/blm_research/data/sampled_tweets_with_toxicity_score.txt'
    get_toxicity_score(sampled_data_path, save_as)