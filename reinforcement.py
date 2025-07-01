
import os 
import gym 
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import os
import random
import ray
import csv
from collections import deque 
import time

import flwr as fl
from flwr.common import Metrics
from logging import INFO, log, DEBUG

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

import torch
from typing import Dict, List, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

import multiprocessing


from model import Net, train, test, get_parameters, set_parameters
from HierFL import HierFL
from client import FlowerClient
from strategy import FedAvgCustom
from load_datasets import load_datasets
from utils import generate_random_sizes




import gc
import gym 
from gym import Env
from gym.spaces import Discrete, Box,MultiDiscrete,Dict,Tuple
import numpy as np

def estimate_energy_max(rounds):
        # Calibrata: E(3) ≈ 500, E(25) ≈ 5000
        a = 5000 / np.log1p(25)   # ≈ 1625.5
        return a * np.log1p(rounds)


class OptHierFLEnv(Env):
    
    def __init__(self, env_name: str,max_round: int = 1,global_rounds: int = 10,history_len: int = 3):
        '''Inizializzazione dell'ambiente di ottimizzazione per HierFL.
        
        -Args:
            env_name: Nome dell'ambiente
            max_round: Numero massimo di round per episodio  (default 1, è un episodio completo)
            global_rounds: Numero di round globali HierFL 
            history_len: Lunghezza della storia degli stati (default 3, per POMDP)


        -Attribites:
            action_space: spazio delle azioni, MultiDiscrete con valori in [5,25] per k1 e 18 [2,10] per k2
            observation_space: spazio delle osservazioni, Box con 3 valori normalizzati tra 0 e 1
                1) accuracy [0,1]
                2) training time [0,1200] 1200 max in quanto >20 min non è accettabile
                5) total energy [0,5000]
                
            
            time_min, time_max: intervallo per il tempo di addestramento
            total_energy_min, total_energy_max: intervallo per l'energia totale

            state_history: deque per tenere traccia delle ultime azioni e ricompense (siamo in POMDP)        
        '''
        # Parametri base
        self.env_name = env_name
        self.max_round = max_round
        self.n_global_rounds = global_rounds
        self.history_len = history_len
    
        
        # Action space: k1 in [5, 25], k2 in [2, 10]
        # We represent this as offsets from the minimum values.
        # k1_range = 25 - 5 + 1 = 21
        # k2_range = 10 - 2 + 1 = 9
        self.action_space=MultiDiscrete([21,9])
        self.observation_space=Box(low=0.0, high=1.0, shape=(3*self.history_len,), dtype=np.float32)

        

        # Range 
        self.acc_min, self.acc_max = 0.0,self.n_global_rounds
        self.time_min, self.time_max = 0, 1200 # Tempo di addestramento in secondi max 
        #self.computation_energy_min, self.computation_energy_max  = 0, 10
        #self.communication_energy_min, self.communication_energy_max = 0, 500
        self.total_energy_min, self.total_energy_max = 0, estimate_energy_max(self.n_global_rounds)  # Energia totale stimata
        #self.comms_min, self.comms_max = 0, 4500

        
        # Stato interno
        self.current_round = 0
        self.state_history = deque(maxlen=self.history_len)
        self.episode_count = 0
        
        
        
        # Inizializza ambiente
        self.reset()

       

    def get_initial_env_state(self):
        '''Restituisce un singolo stato iniziale dell'ambiente, Generato casualmente.

        Returns:
            state (np.ndarray) 3x1: array normalizzato che rappresenta lo stato iniziale 
            dell'ambiente. Contiene 3 valori:
                -  accuratezza iniziale (acc)
                -  tempo di training
                -  energia totale 
        '''
        # Aggiungi randomness per evitare sempre gli stessi valori iniziali
        base_acc = 0.1 + np.random.uniform(0, 0.1)  # [0.1, 0.2]
        base_time = 0.5 + np.random.uniform(0, 0.2)  # [0.5, 0.7]
        base_energy = 0.5 + np.random.uniform(0, 0.2)  # [0.5, 0.7]
        
        return np.array([base_acc, base_time, base_energy], dtype=np.float32)

    
    def normalize(self, value, min_value, max_value):
        ''' Funzione per normalizzare un valore in un range [0, 1].
        -Args:
            value: valore da normalizzare
            min_value: valore minimo del range
            max_value: valore massimo del range
        -Returns:
            value: valore normalizzato in un range [0, 1]
        ''' 
        if max_value - min_value == 0:
            return 0.0
        return np.clip((value - min_value) / (max_value - min_value), 0.0, 1.0)



    
    def denormalize(self, normalized_value, min_value, max_value):
        """Denormalizza un valore.
        -Args:
            normalized_value: valore normalizzato in un range [0, 1]
            min_value: valore minimo del range originale
            max_value: valore massimo del range originale
        -Returns:
            value: valore denormalizzato nel range originale
        """
        return normalized_value * (max_value - min_value) + min_value
    


    def decode_action(self, action):
        '''  Funzione per decodificare l'azione in k1 e k2. 
        -Args:
            action: azione dell'agente con parametri k1 in un range [0, 19] e k2 in un range [0, 9]
        -Returns:
            k1: primo iperparametro, in un range [5, 25]
            k2: secondo iperparametro, in un range [2, 10]
        '''
        k1=int(action[0]+5) 
        k2=int(action[1]+2) 
        return k1, k2


    def step(self,action):
        '''
        La funzione step(action):
        Riceve un'azione dall'agente (qui: una coppia di iperparametri k1, k2)
        Esegue la simulazione con quell'azione (HierFL round)
        Calcola la reward
        Salva i valori su CSV
        Salva i Tempi di esecuzione
        Aggiorna lo stato


        -Args:
            action: azione dell'agente, una coppia di iperparametri k1 e k2
        -Returns:
            state_history: La nuova state history (stato dell'ambiente)
            reward: ricompensa calcolata in base ai risultati della simulazione
            done: flag che indica se la simulazione è terminata (tutti i round completati)
            info: dizionario con informazioni aggiuntive, come i valori di k1 e k2 nell'ultimo round

        '''
        # Inizio del timer lo step
        init_time_step = time.time()  

        #recupero k1 e k2 
        k1, k2 = self.decode_action(action)
        print(f"\nRound {self.current_round + 1}/{self.max_round}: k1={k1}, k2={k2}")
        
        
        # Inizio del timer per l'esecuzione di HierFL
        init_time_run = time.time()  
        results = self.run_HierFL(k1, k2)
        # Fine del timer per l'esecuzione di HierFL
        final_time_run = time.time()  
        # Calcola il tempo impiegato per l'esecuzione di HierFL
        self.run_time = final_time_run - init_time_run  
        

        #Ridefine lo stato dell'ambiente normalizzando i risultati
        self.new_observation = np.array([
            self.normalize(results['accuracy'], self.acc_min, self.acc_max),
            self.normalize(results['training_time'], self.time_min, self.time_max),
            self.normalize(results['total_energy'], self.total_energy_min, self.total_energy_max)
        ], dtype=np.float32)

        # Aggiungi alla storia
        self.state_history.append(self.new_observation)
        
        
        # Inizio del timer per il calcolo della ricompensa
        init_time_reward = time.time()  
        reward = self.compute_reward(self.new_observation)
        # Fine del timer per il calcolo della ricompensa
        final_time_reward = time.time()  
        # Calcola il tempo impiegato per il calcolo della ricompensa
        self.reward_time = final_time_reward - init_time_reward  
        
        

        # Aggiorna stato
        self.current_round += 1
        done = self.current_round >= self.max_round

        

        info = {
            'k1': k1, 
            'k2': k2, 
            'reward': reward,
            'raw_results': results,
            'normalized_state': self.new_observation.tolist()
        }
        
        print(f"--- RL ENV STEP {self.current_round}: Finished. Accuracy: {results['accuracy']:.4f}, Reward: {reward:.4f} ---")
        
        # Scrivi i risultati in un file CSV
         # Inizio del timer per la scrittura del sommario
        init_time_summary = time.time() 
        self.write_summary(k1,k2,reward, self.current_round, results)
         # Fine del timer per la scrittura del sommario
        final_time_summary = time.time() 
        # Calcola il tempo impiegato per scrivere il sommario
        self.summary_time = final_time_summary - init_time_summary  

        # Calcola il tempo impiegato per l'intero step
        final_time_step = time.time()  
        self.round_time = final_time_step - init_time_step  

        # Scrivi i tempi in un file di testo
        self.write_time()

        if self.history_len>=2:
            # Se la storia è abbastanza lunga, restituisci lo stato della storia
            return self._get_history_state(), reward, done, info
        else:
            # Se la storia è troppo corta, restituisci lo stato corrente
            return self.new_observation, reward, done, info

    def write_time(self, time_filename="time.csv"):
        '''Scrive il tempo impiegato per il reset e per ogni round in un file di testo.

        Args:
            time_filename: nome del file in cui salvare i tempi
        -Returns:
            None
        '''
        if not os.path.exists(time_filename):
            with open(time_filename, 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['reset_time', "run_time","reward_time","summary_time",'round_time'])
                f.close()

        if not self.reset_time:
            self.reset_time = 0.0  # Imposta a 0 se non è stato calcolato
        if not self.round_time:
            self.round_time = 0.0
        if not self.run_time:
            self.run_time = 0.0
        if not self.reward_time:
            self.reward_time = 0.0
        if not self.summary_time:
            self.summary_time = 0.0

        # Scrive i tempi nel file CSV
        with open(time_filename, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                self.reset_time,  # Tempo impiegato per il reset
                self.run_time,  # Tempo impiegato per l'esecuzione di HierFL
                self.reward_time,  # Tempo impiegato per il calcolo della ricompensa
                self.summary_time,  # Tempo impiegato per scrivere il sommario
                self.round_time  # Tempo impiegato per l'ultimo round
            ])
            f.close()



    
    def render(self):
        pass


    def reset(self):
        '''Reset dell'ambiente per un nuovo episodio.
        Inizializza i parametri, lo stato e la storia degli stati.
        -Returns:
            state_history_state (np.ndarray) (3*history_len)x1: La storia degli stati 
            inizializzata con lo stato iniziale dell'ambiente
        '''
        
        init_time = time.time()
        
        # Reset contatori
        self.current_round = 0
        self.episode_count += 1

        
       
        gc.collect()  # Forza garbage collection per liberare memoria
        
        
        # Reset history con stati diversificati
        self.state_history.clear()
        for i in range(self.history_len):
            initial_state = self.get_initial_env_state()
            self.state_history.append(initial_state)

        
        #Pulizia memoria
        self._cleanup_memory()
        
        final_time = time.time()
        self.reset_time = final_time - init_time
        
        if self.history_len >= 2:
            return self._get_history_state()
        else:
            return self.state_history[0]
        
    
    def _get_history_state(self):
        '''Restituisce lo stato corrente dell'ambiente come concatenazione della storia degli stati.
        -Returns:
            state_history_state (np.ndarray) (3*history_len)x1: La storia degli stati '''
        return np.concatenate(list(self.state_history))
    
    
    def _cleanup_memory(self):
        """Pulizia della memoria per evitare accumulo di risorse inutilizzate.
        Questo metodo forza la garbage collection e pulisce la cache GPU se disponibile.
        
        -Returns:
            None
        """
        import gc
        import torch
        
        # Forza garbage collection
        gc.collect()
        
        # Pulizia cache GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    

    def compute_reward(self, observation):
        '''Calcola la ricompensa in base ai risultati della simulazione.
        
        -Args:
            observation: osservazione corrente dell'ambiente, un array con 3 valori normalizzati:
                1) accuratezza [0,1]
                2) tempo di training [0,1]
                3) energia totale [0,1]
        -Returns:
                reward: ricompensa calcolata in base ai risultati della simulazione, un valore float
                '''

        # Risultati della simulazione
        results = {
            'accuracy': observation[0],
            'training_time': observation[1],
            'total_energy': observation[2],
        }



        if len(self.state_history)>=2:
            previous_obs=self.state_history[-2]
            delta_acc=results["accuracy"]-previous_obs[0]
            delta_time=previous_obs[1]-results["training_time"]
            delta_energy=previous_obs[2]-results["total_energy"]
        else:
            previous_obs=observation
            # Se non ci sono stati precedenti, non c'è miglioramento 
            delta_acc = 0.0
            delta_time = 0.0
            delta_energy = 0.0
        
        
        # Componenti della reward
        w_acc = 1.5 # Main driver: accuracy  
        w_time = 0.6 # Penalty for time   
        w_energy = 0.9  # Penalty for total energy  

        reward = (w_acc * delta_acc) - \
                (w_time * delta_time) - \
                (w_energy * delta_energy) +\
                (results['accuracy']*w_acc) -\
                (w_energy * results['total_energy']) -\
                (w_time * results['training_time']) 
        

        '''# Bonus per accuracy alta (reward progressivo)
        if results['accuracy'] > 0.7:
            reward += 0.3
        elif results['accuracy'] > 0.5:
            reward += 0.1
        
        # Penalità per performance scarse
        if results["accuracy"] < 0.2:
            reward -= 0.5
        
        #penalità time alti
        if results["training_time"] > 0.6:
            reward-=1  
        '''

        '''T=max(0.01,np.exp(-0.05*self.episode_count))
        exploration_noise=T*np.random.uniform(-0.2,0.2)
        reward+=exploration_noise'''

        return reward


    def write_summary(self,k1,k2, score, step,state,filename="reuslts_trained.csv"):
        '''Scrive i risultati in un file CSV.

        Args:
            score: punteggio ottenuto nell'episodio
            step: numero di step eseguiti
            state: stato finale dell'ambiente
            filename: nome del file CSV in cui salvare i risultati
        '''
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['k1','k2','score', 'step', 'accuracy_tot', 
                                     'training_time_tot', 'computation_energy_tot', 'communication_energy_tot', 
                                     'total_energy_tot', 'communications_tot',
                                     "accuracy_avg",
                                     "training_time_avg",
                                     "computation_energy_avg",
                                     "communication_energy_avg",
                                     "total_energy_avg",
                                     "communications_avg",
                                     "normalized_accuracy",
                                     "normalized_training_time",
                                     "normalized_total_energy"
                                     
                                     #''', 'accuracy_increase','communication_energy_increase'''
                                     ])
                f.close()
       
        # Scrive i risultati nel file CSV            
        with open(filename, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                k1,
                k2,
                score,
                step,
                state["accuracy"],  # accuracy
                state["training_time"],  # training_time
                state["computation_energy"],  # computation_energy
                state["communication_energy"],  # communication_energy
                state["total_energy"],  # total_energy
                state["communications"], # communications
                state["accuracy"]/self.n_global_rounds,
                state["training_time"]/self.n_global_rounds,
                state["computation_energy"]/self.n_global_rounds,
                state["communication_energy"]/self.n_global_rounds,
                state["total_energy"]/self.n_global_rounds,
                state["communications"]/self.n_global_rounds,
                self.normalize(state["accuracy"], self.acc_min, self.acc_max),  # normalized accuracy
                self.normalize(state["training_time"], self.time_min, self.time_max),  # normalized
                self.normalize(state["total_energy"], self.total_energy_min, self.total_energy_max),  # normalized total energy
            ])
            f.close()



            
    def run_HierFL(self, k1, k2):
        '''Esegue un round di HierFL con i parametri k1 e k2.
        -ARGS:
            k1: primo iperparametro, numero di iterazioni locali per client
            k2: secondo iperparametro, numero di aggregazioni tra edge server
        -Returns:
            results: dizionario con i risultati cumulativi della simulazione, contenente:
                - accuracy: accuratezza finale del modello
                - training_time: tempo totale di addestramento in secondi
                - total_energy: energia totale consumata durante l'addestramento
                - computation_energy: energia computazionale (placeholder, non usata)
                - communication_energy: energia di comunicazione (placeholder, non usata)
                - communications: numero di comunicazioni effettuate durante l'addestramento
                
        '''
        print(f"Running HierFL with k1={k1}, k2={k2}")

        args = {
        'EXPERIMENT': 3,
        'SEED': 1,
        'DATASET': "cifar10",
        'DATA_DISTRIBUTION': "IID",
        'TYPE_DISTRIBUTION': "dirichlet",
        'NUM_CLIENTS': 30, # 30
        'LOCAL_ITERATIONS': int(k1), #k1
        'EDGE_AGGREGATIONS': int(k2), # k2 10
        'GLOBAL_ROUNDS': self.n_global_rounds,  # Numero di round globali
        'TRAIN_PHASES': 5,
        #'NUM_EDGES_SERVERS': 5,
        'ACTUAL_GLOBAL_ROUND': 0,
        'ACTUAL_EDGE_ROUND': 0,
        'EDGE_CLIENTS': 10,
        'CLIENT_FRACTION': 0.7,
        'EVALUATE_FRACTION': 0.7,
        'MIN_FIT_CLIENT': 6,
        'MIN_EVALUATE_CLIENT': 6,
        'MIN_AVAILABLE_CLIENT': 8,
        'BATCH_SIZE': 20,
        'LEARNING_RATE': 0.01,
        'EXPONENTIAL_DECAY_RATE': 0.995
        }
        
        
        trainloaders, valloaders, testloader = load_datasets(args)

        args['DEVICE'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Try "cuda" to train on GPU
        print(f"Training on {args['DEVICE']} using PyTorch {torch.__version__} and Flower {fl.__version__}")

        # Specify the resources each of your clients need. 
        # By default, each client will be allocated 1x CPU and 0x GPUs
        client_resources = {"num_cpus": 1, "num_gpus": 0.0}
        args['client_resources'] = client_resources
        if args['DEVICE'].type == "cuda":
            # here we are assigning an entire GPU for each client.
            args['client_resources'] = {"num_cpus": 1, "num_gpus": 1}

        # Leaf directory
        directory = "Experiment_" + str(args['EXPERIMENT'])
        # Parent Directories
        parent_dir = "Experiments"
        # Path
        dir_path = os.path.join(parent_dir, directory)
        args['dir_path'] = dir_path
        # Create the directory 'ihritik'
        try:
            os.makedirs(dir_path, exist_ok=True)
            print("Directory '%s' created successfully" % directory)
        except OSError as error:
            print("Directory '%s' can not be created")
            

        results=HierFL(args, trainloaders, valloaders, testloader)
       

        #Pulizia esplicita dopo ogni run
        self._cleanup_memory()

        return results

    def simulate_hierfl(self, k1, k2):
        """
        Simula l'esecuzione di HierFL con parametri k1, k2.
        Questa funzione simula un round di HierFL e restituisce i risultati come un dizionario.
        -Args:
            k1: primo iperparametro, numero di iterazioni locali per client
            k2: secondo iperparametro, numero di aggregazioni tra edge server
        -Returns:
            results: dizionario con i risultati della simulazione, contenente:
                - accuracy: accuratezza finale del modello
                - training_time: tempo totale di addestramento in secondi
                - total_energy: energia totale consumata durante l'addestramento
                - computation_energy: energia computazionale (placeholder, non usata)
                - communication_energy: energia di comunicazione (placeholder, non usata)
                - communications: numero di comunicazioni effettuate durante l'addestramento
        """
        # Seed diverso per ogni simulazione per evitare risultati identici
        simulation_seed = hash((k1, k2, self.current_round, self.episode_count)) % 10000
        #np.random.seed(simulation_seed)
        
        # Simulazione realistica basata sui parametri
        # Accuracy: migliora con più iterazioni locali ma con diminishing returns
        base_accuracy = 0.3 + (k1 / 100.0) * 0.6  # Base accuracy da k1
        accuracy_noise = np.random.normal(0, 0.05)  # Rumore
        accuracy = np.clip(base_accuracy + accuracy_noise, 0.1, 1.0)
        
        # Training time: cresce con k1 e k2
        base_time = 60 + (k1 * 15) + (k2 * 30)  # Tempo base in secondi
        time_noise = np.random.normal(0, base_time * 0.1)  # 10% di variazione
        training_time = max(10, base_time + time_noise)
        # Total energy: correlato con tempo e parametri
        base_energy = 100 + (k1 * 5) + (k2 * 20) + (training_time * 0.5)
        energy_noise = np.random.normal(0, base_energy * 0.1)
        total_energy = max(50, base_energy + energy_noise)
        
        return {
            'accuracy': accuracy,
            'training_time': training_time,
            'total_energy': total_energy,
            'computation_energy': 0,
            "communication_energy": 0,  
            'communications': 1
        }   
    

if __name__ == "__main__":
    # Inizializza l'ambiente di ottimizzazione per HierFL
    env = OptHierFLEnv(env_name="HierFL-v0", max_round=0, global_rounds=3,history_len=1)  
    

    log_path = os.path.join('Training', 'Logs')

    # PARAMETRI PPO MIGLIORATI
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,    # default SB3, stabile
        n_steps=32,          # Esegui 32 simulazioni prima di un update della policy.
        batch_size=16,       # Suddividi i 32 campioni in mini-batch da 16 per l'update.
        n_epochs=5,         # Ripeti l'update sui dati raccolti per 10 volte
        verbose=1,
        gamma=0.99
    )

    # total_timesteps è ora il NUMERO TOTALE DI SIMULAZIONI che vuoi eseguire.
    # Iniziamo con 250. Se una simulazione dura 2 minuti, il training durerà circa 8.3 ore.
    TOTAL_SIMULATIONS = 250
    print(f"\nAVVIO TRAINING RL per {TOTAL_SIMULATIONS} simulazioni.")
    
    model.learn(total_timesteps=TOTAL_SIMULATIONS, progress_bar=True)

   
 

    # Salva il modello finale
    model.save("ppo_HierFL")
    print("Model saved as ppo_HierFL")
    print("Training completed.")
    # Esegui un test dell'ambiente
    avg_reward,std_reward=evaluate_policy(model, env, n_eval_episodes=10, render=True)
    print(avg_reward)
    print(std_reward)
