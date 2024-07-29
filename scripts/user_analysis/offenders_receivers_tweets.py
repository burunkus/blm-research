import csv
import os
import sys
import networkx as nx
import logging
import logging.handlers
from cdlib import algorithms, readwrite
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


def extract_offenders_and_receivers_tweets(offenders, receivers, data_paths, save_as, allow_overlap):
    logging.info(f"Number of receivers: {len(receivers)}")
    logging.info(f"Number of offenders: {len(offenders)}")
        
    path = '../../../../../zfs/socbd/eokpala/blm_research/data/emotion_analysis_by_users_data/'
    offenders_offensive_and_non_offensive_tweets_data = []
    offenders_offensive_tweets_data = []
    offenders_non_offensive_tweets_data = []
    receivers_offensive_and_non_offensive_tweets_data = []
    receivers_offensive_tweets_data = []
    receivers_non_offensive_tweets_data = []
    
    for data_path in data_paths:
        # Store all tweets in data_path in a dict for fast search
        # tweets_data = defaultdict(list)
        with open(data_path) as csv_file_path_handle:
            csv_reader = csv.reader(csv_file_path_handle, delimiter=',')
            for i, row in enumerate(csv_reader):
                author_id = row[1]
                # tweets_data[author_id] = row
                
                if author_id in offenders:
                    offenders_offensive_and_non_offensive_tweets_data.append(row)
                    # Check if offender's tweet is offensive or non-offensive
                    if row[20] == '1':
                        offenders_offensive_tweets_data.append(row)
                    else:
                        offenders_non_offensive_tweets_data.append(row)
        
                if author_id in receivers:
                    receivers_offensive_and_non_offensive_tweets_data.append(row)
                    # Check if receiver's tweet is offensive or non-offensive
                    if row[20] == '1':
                        receivers_offensive_tweets_data.append(row)
                    else:
                        receivers_non_offensive_tweets_data.append(row)
        
    # Stats
    logging.info(f'Number of offenders offensive and non-offensive tweets: {len(offenders_offensive_and_non_offensive_tweets_data)}')
    logging.info(f'Number of offenders offensive tweets: {len(offenders_offensive_tweets_data)}')
    logging.info(f'Number of offenders non-offensive tweets: {len(offenders_non_offensive_tweets_data)}')
    logging.info(f'Number of receivers offensive and non-offensive tweets: {len(receivers_offensive_and_non_offensive_tweets_data)}')
    logging.info(f'Number of receivers offensive tweets: {len(receivers_offensive_tweets_data)}')
    logging.info(f'Number of receivers non-offensive tweets: {len(receivers_non_offensive_tweets_data)}')

    # Write tweets to file for emotion analysis
    save_as_extention = ''
    if not allow_overlap:
        save_as_extention = '_no_overlap'
    
    with open(path + 'offenders_offensive_and_non_offensive_tweets' + save_as + save_as_extention + '.csv' , mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in offenders_offensive_and_non_offensive_tweets_data:
            csv_handler.writerow(data)
    
    with open(path + 'offenders_offensive_tweets' + save_as + save_as_extention + '.csv' , mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in offenders_offensive_tweets_data:
            csv_handler.writerow(data)
    
    with open(path + 'offenders_non_offensive_tweets' + save_as + save_as_extention + '.csv' , mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in offenders_non_offensive_tweets_data:
            csv_handler.writerow(data)
            
    with open(path + 'receivers_offensive_and_non_offensive_tweets' + save_as + save_as_extention + '.csv' , mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in receivers_offensive_and_non_offensive_tweets_data:
            csv_handler.writerow(data)
    
    with open(path + 'receivers_offensive_tweets' + save_as + save_as_extention + '.csv' , mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in receivers_offensive_tweets_data:
            csv_handler.writerow(data)
            
    with open(path + 'receivers_non_offensive_tweets' + save_as + save_as_extention + '.csv' , mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in receivers_non_offensive_tweets_data:
            csv_handler.writerow(data)        


def extract_offenders_and_receivers(edgelist_paths, data_paths, save_as='', allow_overlap=True):
    
    offenders = []
    receivers = []
    for edgelist_path in edgelist_paths:
        logging.info(f"Loading graph {edgelist_path} ...")
        graph = nx.read_edgelist(edgelist_path, nodetype=str, data=(("weight", int),), create_using=nx.DiGraph())
        logging.info(f"Graph loaded.")
    
        num_of_nodes = len(list(graph.nodes))
        logging.info(f'Number of nodes: {num_of_nodes}')
        num_of_edges = len(list(graph.edges))
        logging.info(f'Number of edges: {num_of_edges}')
    
        # receivers
        for node, in_degree in graph.in_degree:
            if in_degree >= 1:
                receivers.append(node)
        
        # offenders
        for node, out_degree in graph.out_degree:
            if out_degree >= 1:
                offenders.append(node)
    
    if allow_overlap:
        extract_offenders_and_receivers_tweets(set(offenders), 
                                               set(receivers), 
                                               data_paths,
                                               save_as,
                                               allow_overlap)
    else:
        offenders = set(offenders)
        receivers = set(receivers)
        offenders_that_are_not_receivers = offenders - receivers
        receivers_that_are_not_offenders = receivers - offenders
        extract_offenders_and_receivers_tweets(offenders_that_are_not_receivers, 
                                               receivers_that_are_not_offenders, 
                                               data_paths,
                                               save_as,
                                               allow_overlap)


def extract_high_receivers_and_offenders(edgelist_paths, 
                                         data_paths, 
                                         save_as,
                                         threshold, 
                                         threshold_by_mean=False,
                                         allow_overlap=True):
    
    high_offenders = []
    high_receivers = []
    for edgelist_path in edgelist_paths:
        logging.info(f"Loading graph {edgelist_path} ...")
        graph = nx.read_edgelist(edgelist_path, nodetype=str, data=(("weight", int),), create_using=nx.DiGraph())
        logging.info(f"Graph loaded.")
        
        num_offensive_tweets_per_receiver = []
        for node in graph.nodes:
            total_offensive_tweets_received = 0
            for u, v in graph.in_edges(node):
                weight = graph.edges[u, v]['weight']
                total_offensive_tweets_received += weight
            if total_offensive_tweets_received > 0:
                num_offensive_tweets_per_receiver.append((node, total_offensive_tweets_received))
        mean_offensive_tweets_per_receiver = sum([total for node, total in num_offensive_tweets_per_receiver])/len(num_offensive_tweets_per_receiver)
        logging.info(f"The mean number of offensive tweets received per receiver: {mean_offensive_tweets_per_receiver}")
        
        num_offensive_tweets_per_offender = []
        for node in graph.nodes:
            total_offensive_tweets = 0
            for u, v in graph.out_edges(node):
                weight = graph.edges[u, v]['weight']
                total_offensive_tweets += weight
            if total_offensive_tweets > 0:
                num_offensive_tweets_per_offender.append((node, total_offensive_tweets))
        mean_offensive_tweets_per_offender = sum([total for node, total in num_offensive_tweets_per_offender])/len(num_offensive_tweets_per_offender)
        logging.info(f"The mean number of offensive tweets per offender: {mean_offensive_tweets_per_offender}")
        
        for node, total_tweets in num_offensive_tweets_per_receiver:
            if threshold_by_mean:
                if total_tweets > mean_offensive_tweets_per_receiver: 
                    high_receivers.append(node)
            else:
                if total_tweets > threshold:
                    high_receivers.append(node)
        
        for node, total_tweets in num_offensive_tweets_per_offender:
            if threshold_by_mean:
                if total_tweets > mean_offensive_tweets_per_offender:
                    high_offenders.append(node)
            else:
                if total_tweets > threshold:
                    high_offenders.append(node)
        
    if allow_overlap:
        extract_offenders_and_receivers_tweets(set(high_offenders), 
                                               set(high_receivers), 
                                               data_paths,
                                               save_as,
                                               allow_overlap)
    else:
        offenders = set(high_offenders)
        receivers = set(high_receivers)
        high_offenders_that_are_not_receivers = offenders - receivers
        high_receivers_that_are_not_offenders = receivers - offenders
        extract_offenders_and_receivers_tweets(high_offenders_that_are_not_receivers, 
                                               high_receivers_that_are_not_offenders, 
                                               data_paths,
                                               save_as,
                                               allow_overlap)


def main():
    edgelist_paths = [
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2020_reply_tweets_edge_list.edgelist",
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2021_reply_tweets_edge_list.edgelist",
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2022_reply_tweets_edge_list.edgelist"
        ]

    data_paths = ["../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/<file name of 2020 tweets>.csv",
                  "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/<file name of 2021 tweets>.csv", 
                  "../../../../../zfs/socbd/eokpala/blm_research/data/topic_modeling_data/<file name of 2022 tweets>.csv"
                 ]
    
    # Extract the tweets of the offenders that are also receivers and the receivers that are also offenders
    logging.info(f'Processing with overlap ...')
    extract_offenders_and_receivers(edgelist_paths, data_paths, allow_overlap=True)
    logging.info(f'Processing with overlap completed.')
    print()
    
    # Extract the tweets of offenders that are not receivers and receivers that are not offenders
    logging.info(f'Processing without overlap ...')
    extract_offenders_and_receivers(edgelist_paths, data_paths, allow_overlap=False)
    logging.info(f'Processing without overlap completed.')
    
    # Extract high receivers and offenders
    logging.info(f'Processing with high receivers with overlap...')
    # threshold = 50
    threshold = 100
    save_as = f'_high_{threshold}'
    extract_high_receivers_and_offenders(edgelist_paths, 
                                         data_paths,
                                         save_as,
                                         threshold, 
                                         threshold_by_mean=False, 
                                         allow_overlap=True)
    logging.info(f'Processing high receivers with overlap completed.')
    print()
    
    logging.info(f'Processing with high receivers without overlap...')
    extract_high_receivers_and_offenders(edgelist_paths, 
                                         data_paths,
                                         save_as,
                                         threshold, 
                                         threshold_by_mean=False, 
                                         allow_overlap=False)
    logging.info(f'Processing high receivers without overlap completed.')



if __name__ == "__main__":
    log_dir ='./log_folder'
    # custom_name = "_high_receivers_and_offenders_100"
    _ = get_logger(log_dir, get_filename() + custom_name)
    main()