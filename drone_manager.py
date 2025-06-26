class DroneManager():
    def __init__(self, id):
        self.droneId = id
        self.battery_capacity_wh = 98.8       #Watt-hours
        self.battery_capacity_joules = 355680.0  # Convert Wh to Joules (98.8 * 3600)
        self.actual_battery_capacity_J = 355680.0
        self.actual_batteryLevel_percentage = 100.0
        self.num_communications = 0
        self.included = True
        #self.training_time = 0.0
        #self.epochs
        #self.energy_comp_sample = 0.000012544 # Joules MNIST
        #self.train_time_sample = 0.00012544 # seconds MNIST
        self.energy_comp_sample = 0.000049152 # Joules CIFAR10
        self.train_time_sample = 0.00049152 # seconds
        self.channel_capacity = 5.672e6  # 5.672 × 10^6 # Retrieved with Shannon-Hartley Theorem
        self.transmitter_power = 0.5

    def getEnergyLevel(self):
        """
        Check the current battery level.
        :return: The current battery level as a percentage.
        """
        return self.actual_batteryLevel_percentage
    


    def getEnergyCapacity(self):
        """
        Check the current battery level capacity in Joules.
        :return: The current battery level as capacity in Joules.
        """
        return self.actual_battery_capacity_J
    

    def computeEnergyComputation(self, num_samples):
        """ Compute the energy consumed for the computation of data """

        energyCompConsumed = round((num_samples * self.energy_comp_sample), 10)  # Compute total energy consumption for local iteration (# batches x energy for sample)
        #print(f"energy consumed: {energyCompConsumed}")

        return energyCompConsumed
    

    def computeEnergyCommunication(self, size_model_bits):
        """ Compute the energy consumed for the trasmission to edge server of the local model updates """

        communicationLatency = round((size_model_bits / self.channel_capacity), 10)
        energyCommConsumed = round((self.transmitter_power * communicationLatency), 10)
        #print(f"Communication Energy consumed: {energyCommConsumed}")

        return energyCommConsumed
    
    
    
    def computeTrainTimeComputation(self, num_samples):
        """ Compute the training time required for the computation of data """

        training_time = round((num_samples * self.train_time_sample), 10)

        return training_time
    
    
    def getNumCommunications(self):
        """Get the number of communications made during the local update"""

        return self.num_communications


    def setNumCommunications(self, num_communications):
        """ Increase the number of communications with Edge Server"""

        self.num_communications = num_communications
    


    def decreaseEnergyLevel(self, amountEnergyConsumed):
        """
        Decrease the battery level by a certain amount.
        :param amount: The amount to decrease the battery level by.
        :return: None
        """
        #num_samples_batch
        
        remainingBatteryCapacity = round((self.actual_battery_capacity_J - amountEnergyConsumed), 10)
        self.actual_battery_capacity_J = remainingBatteryCapacity

        #print(f"Actual Battery capacity: {self.actual_battery_capacity_J}")
        energyLevelPercentage = round(((self.actual_battery_capacity_J * 100) / self.battery_capacity_joules), 10)
        self.actual_batteryLevel_percentage = energyLevelPercentage
        # if amount < 0:
        #     print("Invalid amount. The amount to decrease must be positive.")
        #     return
        
        # if self.batteryLevel < 10:
        #     print("ERROR! ")

        # if self.current_battery_level - amount < 0:
        #     self.current_battery_level = 0
        # else:
        #     self.current_battery_level -= amount

        #print(f"Battery decreased by {amount}%. Current level: {self.current_battery_level}%")



    def decreaseEnergyLevelCommunication(self, updatedParamsBits):
        """
        Decrease the battery level by a certain amount of consumed energy related for communication parameters.
        :param amount: The amount of bits of updated parameters to decrease the battery level by.
        :return: None
        """
        #num_samples_batch
        communicationLatency = round((updatedParamsBits / self.channel_capacity), 5)
        commEnergyConsumed = self.transmitter_power * communicationLatency
        print(f"Communication Energy consumed: {commEnergyConsumed}")
        remainingBatteryCapacity = self.actual_battery_capacity_J - commEnergyConsumed - 10000
        self.actual_battery_capacity_J = remainingBatteryCapacity

        num_communications = self.num_communications
        self.num_communications = num_communications + 1
        print(f"Actual Battery capacity: {self.actual_battery_capacity_J}")
        energyLevelPercentage = round(((self.actual_battery_capacity_J * 100) / self.battery_capacity_joules), 2)
        print(f"Actual Battery Level: {energyLevelPercentage}")
        self.actual_batteryLevel_percentage = energyLevelPercentage