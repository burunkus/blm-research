import ndjson
import datetime
import sys
import os
import csv
import logging
import logging.handlers
from dateutil import parser


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


def extract_posts_and_comments(file_path, blm_hashtags, keywords, save_path):
    
    parler_data_files = sorted(os.listdir(file_path))
    
    with open(save_path + 'parler_blm.csv', 'w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for parler_data_file in parler_data_files:
            with open(file_path + parler_data_file) as file_handle:
                logging.info(f'Processing {parler_data_file}')
                reader = ndjson.reader(file_handle)
                for i, line in enumerate(reader):
                    if 'createdAt' in line and 'creator' in line and 'id' in line and 'hashtags' in line:
                        date = parser.parse(line['createdAt']).date().strftime('%Y-%m-%d')
                        creator = line['creator']
                        post_id = line['id']
                        comment = line['body']
                        hashtags = line['hashtags']
                        hashtags = {tag.lower() for tag in hashtags}

                        # If hashtags is not empty
                        if hashtags:
                            common_hashtags = hashtags & blm_hashtags
                            # If hashtags contains a blm hashtag
                            if common_hashtags:
                                row = [post_id, creator, date, comment]
                                csv_handler.writerow(row)
                            else:
                                # Check if the comment contains any of the blm keywords
                                for keyword in keywords:
                                    if keyword in comment:
                                        row = [post_id, creator, date, comment]
                                        csv_handler.writerow(row)
                                        # Found a match
                                        break
                        else:
                            pass


def main():
    file_location = "../../../../../../zfs/socbd/eokpala/blm_research/data/parler_data/"
    save_path = "../../../../../../zfs/socbd/eokpala/blm_research/data/"
    hashtags = {"#blm", 
                "#blacklivesmatter", 
                 "#atlantaprotests", 
                 "#kenoshaprotest",
                 "#minneapolisprotest",
                 "#changethesystem",
                 "#justiceforgeorgefloyd",
                 "#georgefloyd",
                 "#floyd",
                 "#breonnataylor",
                 "#justiceforbreonnataylor",
                 "#breonna",
                 "#justiceforjacobblake",
                 "#jacobblake",
                 "#justiceforahmaud",
                 "#ahmaudarbery",
                 "#ahmaud"
               }
    keywords = {"black lives matter", "george floyd", "breonna taylor", "ahmaud arbery"}
    extract_posts_and_comments(file_location, hashtags, keywords, save_path)
    

if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
    main()