import csv
import os
import random
import matplotlib.pyplot as plt
from bertopic import BERTopic
from umap import UMAP
from collections import defaultdict
from collections import Counter


def get_word_intrusion_dataset(model_path, save_dataset_as, save_intruder_as, seed, num_topics=10, num_top_words=5):
    '''
    Generates 10 random topics from the top 50 topics of each years's topic model, selects the top 5
    most probable words in each topic and adds an intruder to the top 5 most probable words. 
    '''
    topic_model = BERTopic.load(model_path)
    topics_info_df = topic_model.get_topic_info()
    default_all_topics = topics_info_df['Topic'].values
    # Ensure topic "-1" is not included in the list
    all_topics = default_all_topics[1:].tolist()
    # Only consider the first 50 topics
    all_topics = all_topics[:50]
    
    random.seed(seed)
    # Randomly sample 10 topics from all model topics
    topics = random.sample(all_topics, num_topics)
    
    # Select the five most probable words from each of the topics
    topic_dict = defaultdict(dict)
    for topic in topics:
        curr_dict = {}
        top_n_words_in_topic = topic_model.get_topic(topic)
        top_five_words = top_n_words_in_topic[:num_top_words]
        
        top_five_words_list = []
        for word, prob in top_five_words:
            top_five_words_list.append(word)
        print(f'Only top 5 words in topic: {topic}\n {top_five_words_list}')
        
        # Pick some other topic
        new_topic = random.choice(all_topics)
        while new_topic == topic:
            new_topic = random.choice(all_topics)
        
        # From the picked topic select a high probability word
        words_in_new_topic = []
        for word, prop in topic_model.get_topic(new_topic):
            words_in_new_topic.append(word)
        
        print(f'Words in new topic: {new_topic}: \n{words_in_new_topic}')
        intruder = random.choice(words_in_new_topic[:num_top_words])
        print(f'Intruder: {intruder}')
        
        # merge intruder with the top 5 words in the current top and shuffle
        merged_list = top_five_words_list + [intruder]
        random.shuffle(merged_list)
        curr_dict['top_words_with_intruder'] = merged_list
        curr_dict['intruder'] = intruder
        topic_dict[topic] = curr_dict
    
    print(f'Topics and most probable words mixed with intruder: \n{topic_dict}')
    # Write datasets to file
    intruders = []
    with open(save_dataset_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for topic, topic_data in topic_dict.items():
            intruders.append((topic, topic_data['intruder']))
            csv_handler.writerow([topic] + topic_data['top_words_with_intruder'])

    with open(save_intruder_as, mode='w') as csv_file_handle:
        csv_handler = csv.writer(csv_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for topic, intruder in intruders:
            csv_handler.writerow([topic] + [intruder])
            
            
def calculate_word_intrusion(dataset, intruder_path):
    '''
    Calculate the word intrusion of a topic model
    Args:
        dataset (String): Path the file containing the intruders identified by users. Each row in dataset is a topic and each column is a user identification of the intruder in a topic. An intruder is a word that is out of place with other words. A word that when removed, all other words become more coherent. 
        intruder_path (String): Path to the file containing the true intruders in each topic
    Returns:
        model_precision_result (Dict): A dictionary of precision result of each topic in dataset. The key is the topic number and value is the precision value 
    '''
    
    correct_intruder = {}
    with open(intruder_path) as input_file_handle:
        csv_reader = csv.reader(input_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader):
            topic = line[0]
            correct_intruder[topic] = line[1]
    
    model_precision_result = {}
    with open(dataset) as input_file_handle:
        csv_reader = csv.reader(input_file_handle, delimiter=',')
        for i, line in enumerate(csv_reader):
            # Skip the column titles
            if i == 0:
                continue
                
            topic = line[0]
            intruders_chosen_by_users = line[1:]
            num_annotators = len(intruders_chosen_by_users)
            
            intruder = [correct_intruder[topic]] * num_annotators
            # See function calc_wi_precision() in  https://github.com/Kiminaka/topic_model_intrusion_eval/blob/master/topic_model_intrusion_eval/evaluation_function.py
            topic_precision = Counter([x == y for x, y in zip(intruders_chosen_by_users, intruder)])[True]/num_annotators
            model_precision_result[topic] = topic_precision
    
    print(f'{dataset} MPs: {model_precision_result}')
    return model_precision_result
    

def plot_word_intrusion(model_precision_results):
    '''
    Plot the box plot of the model precision result of each year's topic model
    Args:
        model_precision_results (List[Dict]): A list of dictionary where each dictionary represents the results of each year's topic model precision. The dictionary key is a topic number and value is the precision value
    Returns:
        None
    '''
    
    labels = ['2020', '2021', '2022']
    
    all_data = []
    for precision_result in model_precision_results:
        all_data.append(sorted(precision_result.values()))
    
    # Plot model precision box plot
    # data = sorted(model_precision_result.values())
    fig, ax = plt.subplots()
 
    # Rectangular box plot
    bplot = ax.boxplot(all_data,
                       vert=True,
                       patch_artist=True,
                       labels=labels)
    
    # Fill with colors
    colors = ['lightgrey', 'lightgrey', 'lightgrey']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Make the median line black
    for median in bplot['medians']:
        median.set_color('red')
    
    # Add horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xlabel('Topic years')
    ax.set_ylabel('Model Precision')
    
    fig.savefig('topic_model_precision_box_plot.pdf', format='pdf', bbox_inches='tight')
    
    # show plot
    plt.show()
    

def handle_dataset(curr_class="offensive"):
    saved_model_path = "../../../../../zfs/socbd/eokpala/blm_research/models/topic_models/"
    model_name = f"topic_modeling_{curr_class}_compliant_seed42"
    
    years = ["2020", "2021", "2022"]
    seed = 23 #5 # 42
    for year in years:
        model_path = saved_model_path + '/' + year + '/' + curr_class + '/' + model_name
        save_dataset_as = year + '_' + curr_class + '_' + str(seed) + '.csv'
        save_intruder_as = year + '_' + curr_class + '_intruder_' + str(seed)+ '.csv'
        get_word_intrusion_dataset(model_path, save_dataset_as, save_intruder_as, seed)
        
        
def handle_word_intrusion(curr_class="offensive"):
    true_intruders_files = [f"2020_{curr_class}_intruder_23.csv", 
                            f"2021_{curr_class}_intruder_23.csv", 
                            f"2022_{curr_class}_intruder_23.csv"]
    
    labeled_files = [f"users_label_2020_{curr_class}.csv", f"users_label_2021_{curr_class}.csv", f"users_label_2022_{curr_class}.csv"]
    model_precision_results = []
    for i, users_file in enumerate(labeled_files):
        true_intruders_file = true_intruders_files[i]
        precision = calculate_word_intrusion(users_file, true_intruders_file)
        model_precision_results.append(precision)
        
    plot_word_intrusion(model_precision_results)
    

def main():
    #handle_dataset()
    #handle_offensive_word_intrusion()
    
    # Non-offensive topics
    # handle_dataset(curr_class="non_offensive")
    # handle_word_intrusion(curr_class="non_offensive")
    
    
if __name__ == "__main__":
    main()
    
