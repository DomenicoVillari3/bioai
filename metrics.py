from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics

#from main import args
from model import Net, set_parameters, test

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    #print(f"evaluate_metrics_aggregation_fn: weighted_average function called")
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy = sum(accuracies) / sum(examples)
    print(f"evaluate_metrics_aggregation_fn: weighted_average accuracy {accuracy}")
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": accuracy}



# The `evaluate` function will be by Flower called after every round
# evaluate aggregated model parameters on the server-side
def evaluate(
        server_round: int, 
        parameters: fl.common.NDArrays, 
        config: Dict[str, fl.common.Scalar],) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    # net = Net().to(args.DEVICE)
    #valloader = valloaders[0]
    # set_parameters(net, parameters)  # Update model with the latest parameters
    #loss, accuracy = test(net, testloader)
    print(f"Server Round: {server_round}")
    #print(f"evaluate_fn: Server-side evaluation loss {loss} / accuracy {accuracy}")
    
    #log the running loss
    # global_writer.add_scalar(
    #     'Global training loss',
    #     loss,
    #     (ACTUAL_GLOBAL_ROUND * EDGE_ROUND) + server_round
    #     )
    
    # global_writer.add_scalar(
    #     'Global accuracy',
    #     accuracy,
    #     (ACTUAL_GLOBAL_ROUND * EDGE_ROUND) + server_round
    #     )
    #return loss, {"accuracy": accuracy}