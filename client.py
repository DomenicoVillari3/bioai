import flwr as fl
import torch

from model import Net, train, test, get_parameters, set_parameters, compute_updated_size



class FlowerClient(fl.client.NumPyClient):
    #def __init__(self, net, trainloader, valloader, cid, clusterid: int):
    def __init__(self, net, trainloader, valloader, droneManager, cid):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.droneManager = droneManager
        #self.clusterid = clusterid

        #self.writer = SummaryWriter("runs/exp1/Cluster "+str(self.clusterid)+"/client_"+cid) # nel client al posto di exp1 si potrebbe mettere il cid




    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)
    


    
    def fit(self, parameters, config):  #set the parameters we received from the server
        # Read values from config
        # server_round = config["server_round"]
        hyperparameters = {
            'lr': config['learning_rate'],
            'exp_decay_rate': config['exponential_dacey_rate'],
            'local_iterations': config['local_iterations']
        }

    
        # Save the initial state of the model (before local training)
        initial_state = {key: value.clone() for key, value in self.net.state_dict().items()}
        set_parameters(self.net, parameters)
        localUpdateEnergyComputation, localUpdateTrainTime = train(self.net, self.trainloader, self.droneManager, hyperparameters, verbose=True)
        
        # Save the updated state of the model (after local training)
        updated_state = {key: value.clone() for key, value in self.net.state_dict().items()}

        # Calculate the size in bits of the local updates (assuming 32-bit precision)
        update_size_bits = compute_updated_size(initial_state, updated_state, precision_bits=32)

        print(f"Local update size: {update_size_bits / 8 / 1e6} MB ({update_size_bits} bits)")
        # diminuisco la batteria perché li manda
        communicationEnergyComputation = self.droneManager.computeEnergyCommunication(update_size_bits)
        #print(f"Communication Energy consumed: {communicationEnergyComputation}")
        self.droneManager.setNumCommunications(1)
        droneCommunications = self.droneManager.getNumCommunications()
        
        return get_parameters(self.net), len(self.trainloader), {"droneId": self.droneManager.droneId,
                                                                 "consumedEnergyComputation": localUpdateEnergyComputation,
                                                                 "trainTimeComputation": localUpdateTrainTime,
                                                                 "consumedEnergyCommunication": communicationEnergyComputation,
                                                                 "num_communications": droneCommunications} 
    
    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
        
