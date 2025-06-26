from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

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

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


# Define a custom strategy extending FedAvg
class FedAvgCustom(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_weights = None  # To store the final model weights
        self.total_samples = 0
        self.drones_info = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        # parameters_aggregated = parameters_to_ndarrays(parameters_aggregated)
        self.final_weights = parameters_aggregated
        print(f"Paramters Aggregated : {self.final_weights.tensor_type}")

        # Total number of samples
        total_samples = sum([num_examples for _, num_examples in weights_results])
        self.total_samples = total_samples
        print(f"Total samples: {self.total_samples}")

        drones_metrics = [
             fit_res.metrics
            for _, fit_res in results
        ]
        print(drones_metrics)

        for drone in drones_metrics:
            droneId = drone['droneId']
            #print(f"droneId: {droneId}")
            if droneId not in self.drones_info:
                self.drones_info[droneId] = drone
            else:
                self.drones_info[droneId]['consumedEnergyComputation'] += drone['consumedEnergyComputation']
                self.drones_info[droneId]['trainTimeComputation'] += drone['trainTimeComputation']
                self.drones_info[droneId]['consumedEnergyCommunication'] += drone['consumedEnergyCommunication']
                self.drones_info[droneId]['num_communications'] += drone['num_communications']

        

        for drone_info_id in self.drones_info:
            print(f"DroneId: {drone_info_id} - consumedEnergyComputation: {self.drones_info[drone_info_id]['consumedEnergyComputation']}")
        #return aggregated_weights, aggregated_metrics
        metrics_aggregated = {}

        return self.final_weights, metrics_aggregated
    
