import json
import sys
from utils import get_next_token


def main():
    file_location = "../../../../zfs/socbd/eokpala/blm_research/data/tweets_2022.jsonl"
    next_token = get_next_token(file_location)
    print(next_token)
    
    
if __name__ == "__main__":
    main()