#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import datetime
import sys
from utils import count_tweets, view_tweets, get_next_token
from twarc import Twarc2, expansions

"""
Tutorial examples:
https://github.com/twitterdev/getting-started-with-the-twitter-api-v2-for-academic-research/blob/main/modules/6a-labs-code-academic-python.md

Tutorials:
https://twarc-project.readthedocs.io/en/latest/tutorials/
"""

# Your bearer token here
TOKEN = "Your Twitter API bearer token"
client = Twarc2(bearer_token=TOKEN, metadata=True)

def get_tweets(save_as, next_token=None):

    # Start and end times must be in UTC
    start_time = datetime.datetime(2022, 1, 1, 0, 0, 0, 0, datetime.timezone.utc)
    end_time = datetime.datetime(2022, 10, 27, 0, 0, 0, 0, datetime.timezone.utc)
    
    query = '(#BLM OR #BlackLivesMatter OR #AtlantaProtest OR #KenoshaProtests OR #MinneapolisProtest OR #ChangeTheSystem OR #JusticeForGeorgeFloyd OR #GeorgeFloyd OR #Floyd OR #AhmaudArbery OR #JusticeForAhmaud OR #Ahmaud OR #BreonnaTaylor OR #JusticeForBreonnaTaylor OR #Breonna OR #JusticeForJacobBlake OR #JacobBlake OR "black lives matter" OR "Ahmaud Arbery" OR "Breonna Taylor" OR "Jacob Blake" OR "George Floyd") -is:retweet lang:en'
    
    print(f"Searching for \"{query}\" tweets from {start_time} to {end_time}...")
    #file_name = "tweets.jsonl"
    file_name = save_as
    
    if next_token:
        # Continue collection from where you stopped previously. You must specify a next_token which
        # can be obtained from the last tweet json collected.
        search_results = client.search_all(query=query, start_time=start_time, end_time=end_time, max_results=100, next_token=next_token)
    else:
        # Search_results is a generator, max_results is max tweets per page, not total, 100 is max when using all expansions.
        search_results = client.search_all(query=query, start_time=start_time, end_time=end_time, max_results=100)
        
    # Get all results page by page:
    for page in search_results:
        # The Twitter API v2 returns the Tweet information and the user, media etc. separately
        # so we use expansions.flatten to get all the information in a single JSON
        result = expansions.flatten(page)

        # Do something with the page of results:
        with open(file_name, "a+") as file_handler:
            for tweet in result:
                file_handler.write(f'{json.dumps(tweet)}\n')

    
if __name__ == "__main__":
    save_as = "../../../../zfs/socbd/eokpala/blm_research/data/tweets_2022.jsonl"
    next_token = "b26v89c19zqg8o3fpywol1drf47oexuznlpxg8um0lev1" 
    # Collect tweets
    get_tweets(save_as, next_token=next_token)
    print("Completed.")
