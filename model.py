import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    #CIFAR 10
    '''
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    '''
    
        
    

    # MNIST
    def __init__(self):
         super().__init__()
         # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
         self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
         # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
         self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
         # Fully connected layer: 1024 input features (64*4*4 after pooling), 512 output features
         self.fc1 = nn.Linear(64 * 4 * 4, 512)
         # Output layer: 512 input features, 10 output features (for 10 MNIST classes)
         self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
         # Apply the first convolution, ReLU, and max pooling
         x = F.relu(self.conv1(x))
         x = F.max_pool2d(x, 2)
       # Apply the second convolution, ReLU, and max pooling
         x = F.relu(self.conv2(x))
         x = F.max_pool2d(x, 2)
         # Flatten the tensor
         x = x.view(-1, 64 * 4 * 4)
         # Apply the fully connected layer with ReLU
         x = F.relu(self.fc1(x))
         # Apply the output layer
         x = self.fc2(x)
         return x
    
    # MNIST LIGHT VERSION

        # First convolutional layer: 1 input channel, 10 output channels, 5x5 kernel
    #     self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
    #     # Second convolutional layer: 10 input channels, 20 output channels, 5x5 kernel
    #     self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     # Fully connected layer: 20*4*4 input features, 50 output features
    #     self.fc1 = nn.Linear(320, 50)
    #     # Output layer: 50 input features, 10 output features (one for each class)
    #     self.fc2 = nn.Linear(50, 10)

    # def forward(self, x):
    #     x = self.conv1(x)                                       # First Convolution
    #     x = F.max_pool2d(x, 2)                                  # Max Pooling
    #     x = F.relu(x)                                           # Relu
    #     x = self.conv2(x)                                       # Second Convolution
    #     x = self.conv2_drop(x)                                  # Droput
    #     x = F.max_pool2d(x, 2)                                  # Max Pooling
    #     x = F.relu(x)                                           # Relu
    #     x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])        # Flatten
    #     x = self.fc1(x)                                         # Fully Connected
    #     x = F.relu(x)                                           # Relu
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)                                         # Output layer
    #     return x
    

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


#def train(net, trainloader, writer, server_round, edge_round, global_round, clusterid, epochs: int, verbose=False):
def train(net, trainloader, droneManager, hyperparameters, verbose=False):
    """Train the network on the training set."""

    criterion = torch.nn.CrossEntropyLoss()

    #optimizer = torch.optim.Adam(net.parameters())
    optimizer = torch.optim.SGD(
        params = net.parameters(), 
        lr = hyperparameters['lr'], 
        momentum = 0.9
        )
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyperparameters['exp_decay_rate'])
    net.train()

    total_time = 0.0

    itered_num = 0
    num_iter = hyperparameters['local_iterations']
    end = False
    loss = 0.0
    correct = 0
    total = 0
    actual_epoch = 0

    local_update_energy_consumed = 0.0
    local_update_train_time = 0.0

    # for the local training, the local iterations are the epoch
    # for epoch in range(10):
    #     correct, total, epoch_loss = 0, 0, 0.0
    #     for batch in trainloader:
    #         images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
    #         optimizer.zero_grad()
    #         outputs = net(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         # Metrics
    #         epoch_loss += loss
    #         num_samples_batch = labels.size(0)
    #         total += labels.size(0)
    #         correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    #         batchEnergyConsumed = droneManager.computeEnergyComputation(num_samples_batch) 
    #         batchTrainTime = droneManager.computeTrainTimeComputation(num_samples_batch)
            
    #         local_update_energy_consumed += batchEnergyConsumed
    #         local_update_train_time += batchTrainTime
    #     epoch_loss /= len(trainloader.dataset)
    #     epoch_acc = correct / total
    #     if verbose:
    #         print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    #     exp_lr_scheduler(actual_epoch, scheduler, 1)
    
    for epoch in range(1000):
        for i, batch in enumerate(trainloader):
            if "img" in batch:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            else:
                images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            #print(f"len labels for batch {i}: {len(labels)}")

            #STARTING TIME
            #start_time = time.time()

            optimizer.zero_grad()
            outputs = net(images)
            batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            # END TIME
            #end_time = time.time()

            #Calculate time for this iterarion
            #iteration_time = end_time - start_time
            #total_time += iteration_time

            # Metrics
            loss += batch_loss
            num_samples_batch = labels.size(0) # the number of samples used
            total += labels.size(0) # the number of samples used
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            # FINE ESECUZIONE BATCH
            
            # computation of energy consumption and training time for batch
            batchEnergyConsumed = droneManager.computeEnergyComputation(num_samples_batch) 
            batchTrainTime = droneManager.computeTrainTimeComputation(num_samples_batch)
            local_update_energy_consumed += batchEnergyConsumed
            local_update_train_time += batchTrainTime

            itered_num += 1
            if itered_num >= num_iter:
                end = True
                print(f"Number of iteration: {itered_num}")
                actual_epoch = epoch + 1
                #exp_lr_scheduler(actual_epoch, scheduler, 1)
                break
        if end: break
        actual_epoch = epoch + 1
        exp_lr_scheduler(actual_epoch, scheduler, 1)
    
    print(f"local update energy consumed: {local_update_energy_consumed}")
    print(f"train time: {local_update_train_time}")

    return local_update_energy_consumed, local_update_train_time


def exp_lr_scheduler(epoch, scheduler, lr_decay_epoch):
    if (epoch+1) % lr_decay_epoch:
        return None
    #print("Learning rate decay")
    scheduler.step()
    return None
    


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            if "img" in batch:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            else:
                images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            #print(labels)
            outputs = net(images)
            #print(outputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# Function to compute the size of the updated model parameters
def compute_updated_size(original_state, updated_state, precision_bits=32):
    total_bits = 0

    # Calculate the number of bits for the parameters that have changed
    for orig_param, updated_param in zip(original_state.values(), updated_state.values()):
        if not torch.equal(orig_param, updated_param):
            # Compute the number of elements (parameters)
            num_elements = orig_param.numel()
            # Calculate total bits
            total_bits += num_elements * precision_bits

    return total_bits