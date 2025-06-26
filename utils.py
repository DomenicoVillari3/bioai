import random
import numpy as np
from collections import Counter
from datetime import datetime
import os

def generate_random_sizes(tot_clients, num_clusters, min_size):
    sizes = [min_size] * num_clusters
    remaining_elements = tot_clients - sum(sizes)

    for i in range(remaining_elements):
        idx = random.choice(range(num_clusters))
        sizes[idx] += 1

    random.shuffle(sizes)

    return sizes



def show_distribution(dataloader, args, train_args, type: str):
    """
    show the distribution of the data on certain client with dataloader
    return:
        percentage of each class of the label
    """
    # if args['DATASET'] == 'mnist':
    #     try:
    #         labels = dataloader.dataset.dataset.train_labels.numpy()
    #     except:
    #         print(f"Using test_labels")
    #         labels = dataloader.dataset.dataset.test_labels.numpy()
    #     # labels = dataloader.dataset.dataset.train_labels.numpy()
    # elif args['DATASET'] == 'cifar10':
    #     try:
    #         labels = dataloader.dataset.dataset.train_labels
    #     except:
    #         print(f"Using test_labels")
    #         labels = dataloader.dataset.dataset.test_labels
    #     # labels = dataloader.dataset.dataset.train_labels
    # elif args['DATASET'] == 'fsdd':
    #     labels = dataloader.dataset.labels
    # else:
    #     raise ValueError("`{}` dataset not included".format(args['DATASET']))
    # num_samples = len(dataloader.dataset)
    # # print(num_samples)
    # idxs = [i for i in range(num_samples)]
    # labels = np.array(labels)
    # unique_labels = np.unique(labels)
    # distribution = [0] * len(unique_labels)
    # for idx in idxs:
    #     img, label = dataloader.dataset[idx]
    #     distribution[label] += 1
    # distribution = np.array(distribution)
    # distribution = distribution / num_samples
    # return distribution
    # Collect all labels from the DataLoader
    # Initialize a Counter to count the occurrence of each label
    label_counter = Counter() 

    for batch in dataloader:
        if args['DATASET']=='mnist':
            images, labels = batch["image"], batch["label"]
            label_counter.update(labels.tolist())  # Convert labels to list and update the counter
        elif args['DATASET']=='cifar10':
            imgs, labels = batch["img"], batch["label"]
            label_counter.update(labels.tolist())  # Convert labels to list and update the counter
        else:
            print("ERROR! Dataset is not available")

    # Display the label distribution in textual format
    with open(train_args['file_path'], "a") as file:
        for label, count in sorted(label_counter.items()):
            file.write(f"\t{type}: {count} sample with Label {label}")
        



def generate_random_clusters_conf(args):
    """Function to create the different configurations of the number of clusters and number of clients in each cluster
    at each phase of the HierFL"""

    assert args['GLOBAL_ROUNDS'] > args['TRAIN_PHASES'], "ERROR! The total number of Global Rounds must be higher of the total number of steps in HierFL"

    cluster_sizes_conf = {}

    random_edge_sequence = generate_random_sequence(args['TRAIN_PHASES'] ,possible_values=[3, 4, 5])

    for phase, edges in enumerate(random_edge_sequence):
        print(f"PHASE {phase+1}: Total edges: {edges}")
        min_size = args['NUM_CLIENTS'] // edges - int(0.3 * (args['NUM_CLIENTS'] // edges))
        cluster_sizes = generate_random_sizes(args['NUM_CLIENTS'], edges, min_size)
        cluster_sizes_conf[phase] = cluster_sizes
    print(cluster_sizes_conf)

    return cluster_sizes_conf

        

def generate_random_sequence(phases, possible_values=[3, 4, 5]):
    # Lista per salvare la sequenza di numeri
    sequence = []

    # Il primo numero può essere scelto casualmente senza restrizioni
    previous_number = random.choice(possible_values)
    sequence.append(previous_number)
    
    for _ in range(phases - 1):
        # Rimuovi il numero precedente dalla lista dei possibili valori per favorire la diversità
        next_number = random.choices(
            possible_values, 
            weights=[0.1 if x == previous_number else 0.45 for x in possible_values]
        )[0]
        sequence.append(next_number)
        previous_number = next_number
    
    return sequence


def log_experiment_file(args, train_args):
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as 'Day, Hour:Minute'
    formatted_time = now.strftime("%d-%m-%Y-%H-%M-%S")
    name_file = "Experiment_" + str(formatted_time)

    file_path = os.path.join(args['dir_path'], name_file)
    train_args['file_path'] = file_path

    with open(file_path, "a") as file:
        file.write(f"EXPERIMENT {args['EXPERIMENT']} Log {formatted_time}\n")
        file.write("================================\n")
        file.write("DISTRIBUTED INFRASTRUCTURE CONFIGURATION:\n")
        file.write(f"TOTAL NUMBER OF CLIENTS: {args['NUM_CLIENTS']}\n")
        file.write(f"TOTAL NUMBER OF EDGE SERVER: [3 - 5]\n\n")

        file.write("HIERARCHICAL FEDERATED LEARNING CONFIGURATION:\n")
        file.write(f"CLIENT LOCAL_ITERATIONS (K1): {train_args['LOCAL_ITERATIONS']}\n")
        file.write(f"TOTAL NUMBER OF EDGE_AGGREGATIONS: {train_args['EDGE_AGGREGATIONS']}\n")
        file.write(f"TOTAL NUMBER OF GLOBAL_ROUNDS: {args['GLOBAL_ROUNDS']}\n")
        file.write(f"TRAIN_PHASES: {args['TRAIN_PHASES']}\n\n")

        file.write("FL STRATEGY CONFIGURATION:\n")
        file.write(f"CLIENT FRACTION: {args['CLIENT_FRACTION']}\n")
        file.write(f"EVALUATE_FRACTION: {args['EVALUATE_FRACTION']}\n")
        file.write(f"CLIENT_RESOURCES: {args['client_resources']}\n\n")

        file.write("TRAINING PROCESS CONFIGURATION:\n")
        file.write(f"DATASET: {args['DATASET']}\n")
        file.write(f"DATA DISTRIBUTION: {args['DATA_DISTRIBUTION']}\n")
        if args['DATA_DISTRIBUTION'] == "NON-IID":
            file.write(f"TYPE DISTRIBUTION: {args['TYPE_DISTRIBUTION']}\n")
        file.write(f"BATCH_SIZE: {args['BATCH_SIZE']}\n")
        file.write(f"LEARNING_RATE: {args['LEARNING_RATE']}\n")
        file.write(f"EXPONENTIAL_DECAY_RATE: {args['EXPONENTIAL_DECAY_RATE']}\n")
        file.write("================================\n")
