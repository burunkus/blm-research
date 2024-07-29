import os
import sys
import networkx as nx
import logging
import logging.handlers
from cdlib import algorithms, readwrite


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

def graph_statistics(edgelist_path):
    
    logging.info(f"Loading graph ...")
    graph = nx.read_edgelist(edgelist_path, nodetype=str, data=(("weight", int),), create_using=nx.DiGraph())
    logging.info(f"Graph loaded.")
    
    num_of_nodes = len(list(graph.nodes))
    logging.info(f'Number of nodes: {num_of_nodes}')
    num_of_edges = len(list(graph.edges))
    logging.info(f'Number of edges: {num_of_edges}')
    
    # receivers
    num_receivers = 0
    for node, in_degree in graph.in_degree:
        if in_degree >= 1:
            num_receivers += 1
    logging.info(f'Number of receivers: {num_receivers}({num_receivers/num_of_nodes}%)')
    
    # abusers
    num_abusers = 0
    for node, out_degree in graph.out_degree:
        if out_degree >= 1:
            num_abusers += 1
    logging.info(f'Number of abusers: {num_abusers}({num_abusers/num_of_nodes}%)\n')
    
    return graph, num_receivers, num_abusers


def interactions(graph):
    
    num_of_offensive_tweets_from_abusers_to_receivers = 0
    # Go through each node
    for node in graph.nodes:
        # For each node, go through its out edges
        for u, v in graph.out_edges(node):
            # Sum the weights (# of offensive tweets from u to v) on each edge (u, v)
            num_of_offensive_tweets_from_abusers_to_receivers += graph.edges[u, v]['weight']
    
    num_of_one_off_offensive_tweets = 0
    num_of_not_one_off_offensive_tweets = 0
    num_of_one_off_nodes = 0
    num_of_not_one_off_nodes = 0
    for node in graph.nodes:
        for u, v in graph.out_edges(node):
            weight = graph.edges[u, v]['weight']
            if weight == 1:
                num_of_one_off_nodes += 1
                num_of_one_off_offensive_tweets += weight
            elif weight > 1:
                num_of_not_one_off_nodes += 1
                num_of_not_one_off_offensive_tweets += weight
    
    logging.info(f'Number of one off offensive tweets: {num_of_one_off_offensive_tweets}')
    logging.info(f'Number of not one off offensive tweets: {num_of_not_one_off_offensive_tweets}')
    logging.info(f'Of the {num_of_offensive_tweets_from_abusers_to_receivers} offensive tweets posted by abusers to receivers, {num_of_one_off_offensive_tweets}({num_of_one_off_offensive_tweets/num_of_offensive_tweets_from_abusers_to_receivers})% are one offs and {num_of_not_one_off_offensive_tweets}({num_of_not_one_off_offensive_tweets/num_of_offensive_tweets_from_abusers_to_receivers})% are not one offs\n')

    
def dual_roles(graph, num_receivers, num_abusers):
    # How many receivers are not abusers i.e how many nodes have out_degree == 0
    num_receivers_but_not_abusers = 0
    for node in graph.nodes:
        if graph.out_degree(node) == 0:
            num_receivers_but_not_abusers += 1
    logging.info(f'Number of receivers that are not abusers: {num_receivers_but_not_abusers}({num_receivers_but_not_abusers/num_receivers}% of all receivers)')
    
    # How many receivers are abusers?
    num_receivers_that_are_abusers = 0
    for node in graph.nodes:
        if graph.in_degree(node) >= 1:
            if graph.out_degree(node) >= 1:
                # Then the current node is a receiver that also abuses
                num_receivers_that_are_abusers += 1
    logging.info(f'Number of receivers that are abusers: {num_receivers_that_are_abusers}({num_receivers_that_are_abusers/num_receivers}% of all receivers)')
    
    # How many abusers are receivers?
    num_abusers_that_are_receivers = 0
    for node in graph.nodes:
        if graph.out_degree(node) >= 1:
            if graph.in_degree(node) >= 1:
                # Then the current node is an abuser that also receives
                num_abusers_that_are_receivers += 1
    logging.info(f'Number of abusers that are receivers: {num_abusers_that_are_receivers}({num_abusers_that_are_receivers/num_abusers}% of all abusers)\n')
    

def abusers_reply_abusers(graph, num_abusers):
    
    # How many offensive edges have a reciprocal edge?
    set_of_edges = set(graph.edges)
    num_reciprocal_edges = 0
    for u, v in set_of_edges:
        if (v, u) in set_of_edges:
            #print(f'{(u, v)} <-> {(v, u)}')
            num_reciprocal_edges += 1
    
    logging.info(f'Number of reciprocal offensive edges: {num_reciprocal_edges}({num_reciprocal_edges/len(set_of_edges)}% of offensive edges)')

    num_offensive_tweets_in_reciprocals = 0
    # How many abusive users engage in reciprocals
    num_abusers_who_engage_back = 0
    for node in graph.nodes:
        # If the current node is an abuser
        if graph.out_degree(node) >= 1:
            # Get the out-edges
            for u, v in graph.out_edges(node):
                if (v, u) in set_of_edges:
                    # The current node u has a reciprocal
                    num_abusers_who_engage_back += 1
                    # How big are these reciprocal offensive interactions?
                    num_offensive_tweets_in_reciprocals += graph.edges[u, v]['weight'] + graph.edges[v, u]['weight']
    logging.info(f'Number of abusive users who engage in a conversation where the receiver reciprocates: {num_abusers_who_engage_back}({num_abusers_who_engage_back/num_abusers}% of abusers)')
    
    num_of_offensive_tweets = 0
    for node in graph.nodes:
        # For each node, go through its out edges
        for u, v in graph.edges(node):
            num_of_offensive_tweets += graph.edges[u, v]['weight'] 
            
    logging.info(f'Number of offensive tweets: {num_of_offensive_tweets}')
    logging.info(f'Number of offensive tweets in reciprocals: {num_offensive_tweets_in_reciprocals}({num_offensive_tweets_in_reciprocals/num_of_offensive_tweets})% of offensive tweets\n')
    return num_of_offensive_tweets


def receiver_experience(graph, num_of_offensive_tweets, num_receivers):
    
    # How many receivers receive repeated abuse from the same user
    num_receivers_with_repeated_abuse = 0
    for node in graph.nodes:
        # Node is a receiver
        if graph.in_degree(node) >= 1:
            for u, v in graph.in_edges(node):
                if graph.edges[u, v]['weight'] > 1:
                    num_receivers_with_repeated_abuse += 1
    
    logging.info(f'Number of receivers that experience repeated abuse: {num_receivers_with_repeated_abuse}({num_receivers_with_repeated_abuse/num_receivers}%) of all receivers')
    
    # How many offensive tweets did the most flooded/targeted receiver get
    max_indegree = 0
    max_indegree_node = None
    for node, degree in graph.in_degree:
        if degree > max_indegree:
            print(node, degree)
            max_indegree = degree
            max_indegree_node = node
            
    print(f'Node: {max_indegree_node} was targeted by {max_indegree} abusers')
    
    num_offensive_tweets_of_most_flooded = 0
    for u, v in graph.in_edges(max_indegree_node):
        num_offensive_tweets_of_most_flooded += graph.edges[u, v]['weight']
    
    logging.info(f'The most targeted/flooded receiver/author: {max_indegree_node} received {num_offensive_tweets_of_most_flooded}({num_offensive_tweets_of_most_flooded/num_of_offensive_tweets}%) of offensive tweets\n')
    
    
def main():
    edge_list = [
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2020_reply_tweets_edge_list.edgelist",
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2021_reply_tweets_edge_list.edgelist",
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2022_reply_tweets_edge_list.edgelist"
        ]
    save_graph_as = [
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2020_reply_tweets_graph.graphml",
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2021_reply_tweets_graph.graphml",
        "../../../../../zfs/socbd/eokpala/blm_research/data/network_analysis_data/2022_reply_tweets_graph.graphml"
        ]

    for i, current_year_edge_list in enumerate(edge_list):
        year = current_year_edge_list.split('/')[-1].split('.')[0].split('_')[0]
        logging.info(f'Year: {year}')

        graph, num_receivers, num_abusers = graph_statistics(current_year_edge_list)

        logging.info('Storing the graph in graphml format')
        nx.write_graphml(graph, save_graph_as[i])
        logging.info('Graph stored.')
        
        interactions(graph)
        dual_roles(graph, num_receivers, num_abusers)
        num_of_offensive_tweets = abusers_reply_abusers(graph, num_abusers)
        receiver_experience(graph, num_of_offensive_tweets, num_receivers)
        print()


if __name__ == "__main__":
    log_dir ='./log_folder'
    _ = get_logger(log_dir, get_filename())
    main()