import csv
import os
from bertopic import BERTopic
from umap import UMAP
from topic_modeling_non_offensive_utils import NON_OFFENSIVE_PATH
from topic_modeling_non_offensive_utils import SAVE_MODEL_PATH, SAVE_DOCUMENT_INFO_PATH, CURRENT_YEAR, CURRENT_EXPERIMENT_CLASS 

    
def get_documents(file_path):
    '''
    Get the documents (tweets) from file_path
    Args:
        file_path (String): the location of the file to be used for topic modeling 
    Returns:
        documents (List[Str]): A list of documents where each element is a document (tweet)
    '''
    
    seen = set() 
    with open(file_path) as csv_file_handle:
        csv_reader = csv.reader(csv_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader):
            tweet = line[2]
            seen.add(tweet)
    
    documents = list(seen)
    del seen              
    return documents 


def topic_modeling(documents, save_as):
    '''
    Performs topic modeling using documents
    Args:
        documents (List[Str]): a list of documents 
    Returns:
        all_topics (Dict(int, List[Tuple(str, float)]): A dictionary of topics as keys and value is a list of tuples of
        the most representative words in a topic and the probabilities of the words.
        most_repr_docs (Dict(int, List[String]): A dictionary of topic as key and list of most representative documents as value
    '''
    
    seed = 42
    umap_model = UMAP(n_neighbors=15, 
                      n_components=5, 
                      min_dist=0.0, 
                      metric='cosine', 
                      random_state=seed)
    
    # Use diversity to diversify words in each topic to limit duplicate words that can occur in each topic
    topic_model = BERTopic(language='english', diversity=0.2, umap_model=umap_model, min_topic_size=500)
    
    # Fit the model and return predicted topic per document or tweet and the probability of the assigned topic per document 
    topics, probs = topic_model.fit_transform(documents) 
    
    # Access frequent topics that were generated
    freq = topic_model.get_topic_info()
    freq = freq.to_dict()
    topics_ = list(freq['Topic'].values())
    counts = list(freq['Count'].values())
    topic_names = list(freq['Name'].values())
    print('Frequent topics:')
    print('Topic, Frequency, Name')
    for topic, count, name in zip(topics_, counts, topic_names):
        print(f'{topic}, {count}, {name}')
    
    # Get all topics: -1 refers to all outliers and should typically be ignored
    all_topics = topic_model.get_topics() 
    
    # Get the most representative docs for topics
    most_repr_docs = topic_model.get_representative_docs()
    
    # Get document information
    document_info = topic_model.get_document_info(documents)
    # Write it to file
    folder = SAVE_DOCUMENT_INFO_PATH + CURRENT_YEAR + '/' + CURRENT_EXPERIMENT_CLASS
    if not os.path.exists(folder):
        os.makedirs(folder)
    document_info.to_csv(folder + '/' + save_as, sep=',')
    
    num_tweets = len(topics)
    print(f'Number of tweets/documents: {num_tweets}')
    
    # map each topic to the number of tweets or documents assigned to the topic
    tweets_corresponding_to_each_topic = {}
    for topic in topics:
        if topic in tweets_corresponding_to_each_topic:
            tweets_corresponding_to_each_topic[topic] += 1
        else:
            tweets_corresponding_to_each_topic[topic] = 1
    topic_frequency = sorted(tweets_corresponding_to_each_topic.items())
    
    # Save model
    folder = SAVE_MODEL_PATH + CURRENT_YEAR + '/' + CURRENT_EXPERIMENT_CLASS
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    topic_model.save(folder + '/' + save_as.split('.')[0])
    
    return all_topics, most_repr_docs, topic_frequency, num_tweets


def write_topics_to_csv(all_topics, most_repr_docs, topic_frequency, num_tweets, save_as):
    '''
    Write the discovered topics to a csv file. Each column in the csv file is a topic. 
    Args:
        all_topics (List[List[Tuple(str, float)]]): a list of topics, where each element is a list of one topic containing 
        the words and probabilities of the words in the topic.
        topics (List[int]): The list of topics for csv header
        save_as (String): The name to save the csv file as
    Returns:
        None
    '''
    
    num_topics = len(all_topics)
    num_words_per_topic = len(all_topics[0])
    topics = list(all_topics.keys())
    
    folder = 'results/' + CURRENT_YEAR + '/' + CURRENT_EXPERIMENT_CLASS
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    with open(f'{folder}/' + save_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = []
        for topic in topics:
            header.append("Topic " + str(topic))
            header.append("Prob")
            
        csv_handler.writerow(header)
        for i in range(num_words_per_topic):
            row = []
            for j in range(num_topics):
                topic = topics[j]
                word, prob = all_topics[topic][i]
                row.append(word)
                row.append(prob)
            csv_handler.writerow(row)
        
        # Add some space
        csv_handler.writerow([])
        csv_handler.writerow([])
        
        # Write most representative documents
        header = ["Most repr docs"]
        csv_handler.writerow(header)
        csv_handler.writerow(["Topic", "Documents"])
        for topic, documents in sorted(most_repr_docs.items()):
            for i, document in enumerate(documents):
                if i == 0:
                    csv_handler.writerow([topic, document]) # write topic once
                else:
                    csv_handler.writerow(["", document])
                    
        # Add some space
        csv_handler.writerow([])
        csv_handler.writerow([])
        
        # Write frequency of topics - the count of the documents assigned to a particular topic
        header = ["Topic frequency"]
        csv_handler.writerow(header)
        csv_handler.writerow(["Topic", "Count", "Percentage"])
        for topic, count in topic_frequency:
            csv_handler.writerow([topic, count, round((count/num_tweets)*100, 2)])
            

def main():
    current_file_name = os.path.basename(__file__).split('.')[0]
    if CURRENT_EXPERIMENT_CLASS == 'non_offensive':
        file_path = NON_OFFENSIVE_PATH
        file_name_tokens = current_file_name.split('_')
        new_name = 'topic_modeling_non_offensive_compliant_seed42'
        save_as = new_name + '.csv'
            
    documents = get_documents(file_path)
    all_topics, most_repr_docs, topic_frequency, num_tweets = topic_modeling(documents, save_as)
    write_topics_to_csv(all_topics, most_repr_docs, topic_frequency, num_tweets, save_as)
    

if __name__ == "__main__":
    main()