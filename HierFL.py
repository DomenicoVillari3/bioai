import flwr as fl
from typing import Dict, List, Optional, Tuple
import torch
import time
import random
import numpy as np
import copy

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.strategy.aggregate import aggregate

from strategy import FedAvgCustom
from model import Net, test, get_parameters, set_parameters
from client import FlowerClient
from drone_manager import DroneManager
from metrics import weighted_average, evaluate
from utils import generate_random_sizes, show_distribution, generate_random_clusters_conf, log_experiment_file


def HierFL(args, trainloaders, valloaders, testloader):

    def fit_config(server_round: int):
        """Return training configuration dict for each round.
        """
        print("fit_config function called on configure_fit")

        config = {
            "server_round": server_round,  # The current round of federated learning 
            "local_iterations": train_args['LOCAL_ITERATIONS'],
            "learning_rate": args['LEARNING_RATE'],
            "exponential_dacey_rate": args['EXPONENTIAL_DECAY_RATE']
        }

        return config


    def client_fn_cluster(cid: str):
        
        net = Net().to(args['DEVICE'])

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders_cluster[int(cid)]
        valloader = valloaders_cluster[int(cid)]
        droneManager = copy.deepcopy(clients_cluster[int(cid)])
        return FlowerClient(net, trainloader, valloader, droneManager, cid).to_client()
    

    
    model = Net()
    # Calculate the total number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Assuming 32-bit floating point numbers
    size_bits = num_params * 32

    # Print the size in bits
    #print(f"Model size in bits: {size_bits}")
    
    #name_model_params = args['DATASET'] + '_model_parameters.pth'
    #torch.save(model.state_dict(), name_model_params)
    #model.load_state_dict(torch.load(name_model_params))
    initialisation_params = get_parameters(model)


    global_results = []
    global_training_time = {}
    global_metrics = {}

    #cluster_configurations = generate_random_clusters_conf(args)
    cluster_configurations = {
        0: [6, 6, 7, 5, 6],     # 5 CLUSTERS
        1: [10, 9, 11],         # 3 CLUSTERS
        2: [7, 9, 9, 5],        # 4 CLUSTERS
        3: [12, 9, 9],          # 3 CLUSTERS
        4:[5, 5, 6, 6, 8]       # 5 CLUSTERS
    }

    train_args_conf = [
        {
            'LOCAL_ITERATIONS': args["LOCAL_ITERATIONS"],
            'EDGE_AGGREGATIONS': args["EDGE_AGGREGATIONS"],
        }
    ]

    #train_args_conf = [
        # {
        #     'LOCAL_ITERATIONS': 12,
        #     'EDGE_AGGREGATIONS': 5,
        # },
        # {
        #     'LOCAL_ITERATIONS': 9,
        #     'EDGE_AGGREGATIONS': 6,
        # },
        # {
        #     'LOCAL_ITERATIONS': 8,
        #     'EDGE_AGGREGATIONS': 7,
        # },
        # {
        #     'LOCAL_ITERATIONS': 6,
        #     'EDGE_AGGREGATIONS': 9,
        # },
        # {
        #     'LOCAL_ITERATIONS': 4,
        #     'EDGE_AGGREGATIONS': 13,
        # }
    #]
    

    clients = []
    for id in range(args['NUM_CLIENTS']):
        clients.append(DroneManager(id))


    
    for train_args in train_args_conf:
        log_experiment_file(args, train_args)
        params = initialisation_params
        phase = 0

        cumulative_statistics = {}
        CompEnergyConsumed = 0.0
        CommEnergyConsumed = 0.0
        trainTime = 0.0
        numCommunications = 0

        for global_round in range(args['GLOBAL_ROUNDS']):
            
            # Change cluster configurations after each 5 Global rounds 
            if global_round % args['TRAIN_PHASES'] == 0:
                print(f"PHASE {phase+1} - CONFIGURATION WITH {len(cluster_configurations[phase])} CLUSTERS\n")
                with open(train_args['file_path'], "a") as file:
                    file.write(f"\n----------------------------------------------------------------------------------\n")
                    file.write(f"PHASE {phase+1} - CONFIGURATION WITH {len(cluster_configurations[phase])} CLUSTERS\n")
                
                t_trainloaders = copy.deepcopy(trainloaders)
                t_valloaders = copy.deepcopy(valloaders)
                t_clients = len(clients)

                subset_clients_inf = 0
                subset_clients_sup = 0
                cluster_dataloaders = {}    # data distributed among differents clusters

                for j, size in enumerate(cluster_configurations[phase]):
                    with open(train_args['file_path'], "a") as file:
                        file.write(f"\n\nCluster {j} with size {size}")

                    selected_trainloaders = []
                    selected_valloaders = []

                    selected_trainloaders.extend(t_trainloaders[:size])
                    del t_trainloaders[:size]
                    selected_valloaders.extend(t_valloaders[:size])
                    del t_valloaders[:size]
                    #selected_clients.extend(clients[subset_clients_inf:subset_clients_sup])

                    subset_clients_sup = subset_clients_sup + size
                    print(f"inf: {subset_clients_inf}")
                    print(f"sup: {subset_clients_sup}")
                    num_drones_selected = len(clients[subset_clients_inf:subset_clients_sup])
                    print(f"Number of drones selected: {num_drones_selected}")
                    assert len(selected_trainloaders) == len(selected_valloaders), "ERROR, train and validation dataloaders have different sizes"
                    assert num_drones_selected == len(selected_trainloaders), "ERROR, clients and train dataloaders have different sizes"

                    # VERIFY THE DATA DISTRIBUTION OF EACH CLIENT SELECTED
                    for client_id in range(len(selected_trainloaders)):
                        with open(train_args['file_path'], "a") as file:
                            file.write(f"\nClient {client_id+1} - ")
                        show_distribution(selected_trainloaders[client_id], args, train_args, 'Train Loader')
                        show_distribution(selected_valloaders[client_id], args, train_args, 'Validation Loader')

                    # VERIFY THE CLIENTS SELECTED
                    for drone in clients[subset_clients_inf:subset_clients_sup]:
                        print(f"Drone {drone.droneId} joins in Cluster {j+1}")
                        #del t_clients[subset_clients_inf:subset_clients_sup]
                    
                    #print(f"Remaining clients: {(t_clients - num_drones_selected)}")
                    
                    # dataloaders
                    # cluster_dataloaders has elements equals to the number of clusters in the current configuration 
                    cluster_dataloaders[j+1] = {
                        'train': selected_trainloaders,
                        'validation': selected_valloaders,
                        'clients': clients[subset_clients_inf:subset_clients_sup]
                    }
                    subset_clients_inf = subset_clients_sup

                phase += 1
                print(cluster_configurations)


            # * STARTS DRONES HIERARCHICAL FEDERATED LEARNING
            #########################################################################################################

            #start_time = time.time()
            # with open(train_args['file_path'], "a") as file:
            #     file.write(f"GLOBAL ROUND {global_round+1} is running!")
            print(f"GLOBAL ROUND {global_round+1} is running!")

            CompEnergyConsumedRound = 0.0
            CommEnergyConsumedRound = 0.0
            trainTimeRound = 0.0
            numCommunicationsRound = 0

            for edge in range(len(cluster_dataloaders)):
                trainloaders_cluster = cluster_dataloaders[edge+1]['train']
                valloaders_cluster = cluster_dataloaders[edge+1]['validation']
                clients_cluster = cluster_dataloaders[edge+1]['clients']
                num_clients_cluster = len(trainloaders_cluster)

                strategy_cluster = FedAvgCustom(
                fraction_fit= args['CLIENT_FRACTION'],  # C fraction, meaning 10% of clients are selected each round
                fraction_evaluate= args['EVALUATE_FRACTION'],
                #min_fit_clients= args['MIN_FIT_CLIENT'], # Minimum number of clients used in each round
                #min_evaluate_clients= args['MIN_EVALUATE_CLIENT'],
                #min_available_clients= args['MIN_AVAILABLE_CLIENT'], # Ensure all clients are available for selection
                initial_parameters=fl.common.ndarrays_to_parameters(params),
                on_fit_config_fn=fit_config,
                #evaluate_fn=evaluate  # Pass the evaluation function
                )

                # Run simulation for cluster 1
                fl.simulation.start_simulation(
                    client_fn=client_fn_cluster,
                    num_clients=num_clients_cluster,  # Number of clients in cluster
                    config=fl.server.ServerConfig(num_rounds=train_args['EDGE_AGGREGATIONS']),
                    strategy = strategy_cluster,
                    client_resources= args['client_resources']
                )

                final_weights_cluster = strategy_cluster.final_weights
                total_samples = strategy_cluster.total_samples 
                drones_info = strategy_cluster.drones_info
                global_results.append((parameters_to_ndarrays(final_weights_cluster), total_samples))

                print(f"SIMULATION CLUSTER {edge+1} FINISHED")

                
                #print(drones_info)
                for droneid in drones_info:
                    #print(type(drone['consumedEnergyComputation']))
                    droneId = drones_info[droneid]['droneId']
                    #print(f"droneId: {droneId}")
                    consumedEnergyComputation = drones_info[droneid]['consumedEnergyComputation']
                    CompEnergyConsumedRound += consumedEnergyComputation
                    #print(f"consumedEnergyComputation: {consumedEnergyComputation}")
                    consumedEnergyCommunication = drones_info[droneid]['consumedEnergyCommunication']
                    CommEnergyConsumedRound += consumedEnergyCommunication
                    #print(f"consumedEnergyCommunication: {consumedEnergyCommunication}")
                    clients[droneId].decreaseEnergyLevel(consumedEnergyComputation)
                    clients[droneId].decreaseEnergyLevel(consumedEnergyCommunication)

                    actual_num_communications = clients[droneId].getNumCommunications()
                    updated_num_communications = actual_num_communications + drones_info[droneid]['num_communications']
                    #print(updated_num_communications)
                    clients[droneId].setNumCommunications(updated_num_communications)
                    numCommunicationsRound += drones_info[droneid]['num_communications']
                    trainTimeComputation = drones_info[droneid]['trainTimeComputation']
                    trainTimeRound += trainTimeComputation


            print("RESULTS")
            global_parameters_aggregated = ndarrays_to_parameters(aggregate(global_results))
            global_results.clear()
            global_net = Net().to(args['DEVICE'])
            global_parameters_ndarrays = parameters_to_ndarrays(global_parameters_aggregated)
            params = global_parameters_ndarrays
            set_parameters(global_net, global_parameters_ndarrays)  # Update model with the latest parameters ndarrays
            global_loss, global_accuracy = test(global_net, testloader)


            with open(train_args['file_path'], "a") as file:
                file.write(f"\n\nGLOBAL TEST ROUND: {global_round+1} Evaluation -")
                file.write(f"Loss: {global_loss} \tAccuracy: {global_accuracy} \t")
                file.write(f"Training time: {trainTimeRound}s\n")
                file.write(f"Energy Computation: {CompEnergyConsumedRound} \tEnergy Communication: {CommEnergyConsumedRound} \tNumber of communications: {numCommunicationsRound} ")

            CompEnergyConsumed += CompEnergyConsumedRound
            CommEnergyConsumed += CommEnergyConsumedRound
            trainTime += trainTimeRound
            numCommunications += numCommunicationsRound

            cumulative_statistics[global_round+1] = {
                "Loss": global_loss,
                "Accuracy": global_accuracy,
                "Training time": trainTime,
                "Computation Energy": CompEnergyConsumed,
                "Communication Energy": CommEnergyConsumed,
                "Total Energy": CompEnergyConsumed + CommEnergyConsumed,
                "Communications": numCommunications
            }

        with open(train_args['file_path'], "a") as file:
            file.write(f"\n\nSUMMARY EXPERIMENTS-\n")
            for round in cumulative_statistics:
                # if round+1 == len(cumulative_statistics):
                #     break
                file.write(f"Loss: {cumulative_statistics[round]['Loss']} \tAccuracy: {cumulative_statistics[round]['Accuracy']} \t")
                file.write(f"Training time: {cumulative_statistics[round]['Training time']}s\t")
                file.write(f"Energy Computation: {cumulative_statistics[round]['Computation Energy']} \t \
                            Energy Communication: {cumulative_statistics[round]['Communication Energy']} \t \
                            Total Energy: {cumulative_statistics[round]['Total Energy']} \t \
                            Number of communications: {cumulative_statistics[round]['Communications']}\n ")
                

        # VERIFY TECHNICAL STATISTICS OF DRONES
        with open(train_args['file_path'], "a") as file:
            file.write(f"\n\nSUMMARY DRONE STATISTICS-\n")
            for drone in clients:
                file.write(f"\nDrone {drone.droneId} -\t Battery Level: {drone.actual_batteryLevel_percentage}% -\t Communications with base stations: {drone.num_communications} ")

    #------------------------return final results ------------------------------------------------
    cumulative={"accuracy": 0.0, "loss": 0.0, "training_time": 0.0, 
         "computation_energy": 0.0, "communication_energy": 0.0, 
         "total_energy": 0.0, "communications": 0}
    
    #Itera su tutte le statistiche cumulative per calcolare la somma totale
    for round in cumulative_statistics:
        cumulative['accuracy']+=cumulative_statistics[round]['Accuracy']
        cumulative['loss']+=cumulative_statistics[round]['Loss']
        cumulative['training_time']+=cumulative_statistics[round]['Training time']
        cumulative['computation_energy']+=cumulative_statistics[round]['Computation Energy']
        cumulative['communication_energy']+=cumulative_statistics[round]['Communication Energy']
        cumulative['total_energy']+=cumulative_statistics[round]['Total Energy']
        cumulative['communications']+=cumulative_statistics[round]['Communications']   

    import gc
    gc.collect()

    return cumulative
