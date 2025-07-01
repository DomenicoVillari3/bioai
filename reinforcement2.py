import os
import gym
from gym import Env
from gym.spaces import MultiDiscrete, Box
import numpy as np
import time
import csv

# Importa le tue funzioni di simulazione
from HierFL import HierFL
from load_datasets import load_datasets
import torch

class OptHierFLEnv(Env):
    """
    Ambiente RL ottimizzato per simulazioni costose.
    Ogni step corrisponde a un intero episodio (una singola esecuzione di HierFL).
    """
    def __init__(self, env_name: str, global_rounds: int = 3):
        super(OptHierFLEnv, self).__init__()
        
        self.env_name = env_name
        self.n_global_rounds = global_rounds
        self.episode_count = 0

        # AZIONI: k1 in [5, 25], k2 in [2, 10]
        # k1_range = 25 - 5 + 1 = 21 valori
        # k2_range = 10 - 2 + 1 = 9 valori
        self.action_space = MultiDiscrete([21, 9])

        # STATO (OSSERVAZIONE):
        # Poiché ogni episodio è indipendente, lo stato non contiene informazioni utili
        # per la decisione successiva. Usiamo uno stato fittizio e costante.
        # L'agente imparerà a ignorarlo e si baserà solo sulla ricompensa ottenuta.
        self.observation_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Pesi per la funzione di ricompensa (li mettiamo qui per configurarli facilmente)
        self.w_acc = 1.5
        self.w_time = -0.001   # Negativo perché è un costo
        self.w_energy = -0.002 # Negativo perché è un costo
        
        print("Ambiente RL per ottimizzazione HierFL inizializzato.")
        print(f"Pesi Ricompensa: w_acc={self.w_acc}, w_time={self.w_time}, w_energy={self.w_energy}")

    def decode_action(self, action):
        """Decodifica l'azione dell'agente nei parametri k1 e k2."""
        k1 = int(action[0] + 5)  # Range [5, 25]
        k2 = int(action[1] + 2)  # Range [2, 10]
        return k1, k2

    def step(self, action):
        """
        Esegue un passo che corrisponde a un'intera simulazione.
        """
        self.episode_count += 1
        k1, k2 = self.decode_action(action)
        
        print(f"\n--- INIZIO SIMULAZIONE (Episodio {self.episode_count}): k1={k1}, k2={k2} ---")
        
        # 1. Esegui la simulazione (la parte lenta)
        start_time = time.time()
        results = self.run_HierFL(k1, k2)
        end_time = time.time()
        print(f"--- Simulazione completata in {end_time - start_time:.2f} secondi ---")

        # 2. Calcola lo score finale basato sui risultati ASSOLUTI
        reward = self.compute_final_score(results)

        # 3. L'episodio finisce SEMPRE dopo un passo
        done = True
        
        # Salva un log dettagliato della simulazione
        self.write_summary(k1, k2, reward, self.episode_count, results)
        
        # Lo stato restituito è fittizio, l'agente non lo userà
        next_observation = np.array([0.0], dtype=np.float32)
        
        info = {'k1': k1, 'k2': k2, 'final_score': reward}
        print(f"--- FINE SIMULAZIONE. Score: {reward:.4f} ---")
        
        return next_observation, reward, done, info

    def reset(self):
        """
        Resetta l'ambiente. Si limita a restituire lo stato fittizio iniziale.
        """
        # Non c'è bisogno di pulire la memoria (gc.collect) qui,
        # perché le variabili di `run_HierFL` sono locali e vengono deallocate.
        return np.array([0.0], dtype=np.float32)

    def render(self, mode='human'):
        pass

    def compute_final_score(self, results):
        """
        Calcola uno score singolo per l'intera simulazione.
        Più alto è, meglio è.
        Usa i valori RAW (non normalizzati) per un bilanciamento più chiaro e stabile.
        """
        # Il beneficio principale è l'accuratezza. La moltiplichiamo per 100 per darle più "peso".
        accuracy_benefit = results['accuracy'] * 100
        
        # I costi sono tempo ed energia
        time_cost = results['training_time']
        energy_cost = results['total_energy']
        
        # Lo score è una somma pesata di benefici e costi.
        # I pesi dei costi sono negativi.
        score = (self.w_acc * accuracy_benefit) + \
                (self.w_time * time_cost) + \
                (self.w_energy * energy_cost)

        # RIMUOVIAMO ASSOLUTAMENTE il rumore casuale dalla ricompensa.
        # È fondamentale per la stabilità dell'apprendimento.
        
        return score

    def write_summary(self, k1, k2, score, step, state, filename="reuslts_trained.csv"):
        """Scrive i risultati in un file CSV."""
        # (Questa funzione va bene così com'è, non necessita modifiche)
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            if not file_exists:
                # Scrivi l'header se il file non esiste
                csv_writer.writerow(['k1','k2','score','step', ...]) # Usa il tuo header completo
            
            # Scrivi i dati
            csv_writer.writerow([k1, k2, score, step, ...]) # Usa i tuoi dati completi

    def run_HierFL(self, k1, k2):
        """Esegue un round di HierFL con i parametri k1 e k2."""
        # (Questa funzione va bene così com'è, non necessita modifiche)
        print(f"Esecuzione HierFL con k1={k1}, k2={k2}")
        args = {
            'LOCAL_ITERATIONS': k1,
            'EDGE_AGGREGATIONS': k2,
            # ... tutti gli altri tuoi parametri fissi
            'EXPERIMENT': 3, 'SEED': 1, 'DATASET': "cifar10", 'DATA_DISTRIBUTION': "IID",
            'TYPE_DISTRIBUTION': "dirichlet", 'NUM_CLIENTS': 30, 'GLOBAL_ROUNDS': self.n_global_rounds,
            'TRAIN_PHASES': 5, 'EDGE_CLIENTS': 10, 'CLIENT_FRACTION': 0.7,
            'EVALUATE_FRACTION': 0.7, 'MIN_FIT_CLIENT': 6, 'MIN_EVALUATE_CLIENT': 6,
            'MIN_AVAILABLE_CLIENT': 8, 'BATCH_SIZE': 20, 'LEARNING_RATE': 0.01,
            'EXPONENTIAL_DECAY_RATE': 0.995
        }
        trainloaders, valloaders, testloader = load_datasets(args)
        args['DEVICE'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = HierFL(args, trainloaders, valloaders, testloader)
        return results
    

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    # 1. Inizializza il nuovo ambiente
    # Puoi cambiare i global_rounds per test più veloci, es. global_rounds=1
    env = OptHierFLEnv(env_name="HierFL-v0-single-step", global_rounds=3)
    
    # (Opzionale ma consigliato) Controlla che l'ambiente sia valido
    check_env(env)

    # 2. Configura PPO per essere sample-efficient
    # Questi iperparametri dicono a PPO di eseguire poche simulazioni
    # e di imparare il più possibile da esse.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # Un po' più alta per imparare più in fretta
        n_steps=32,          # Esegui 32 simulazioni prima di un update della policy.
                             # Un valore più basso (es. 20) rende l'update più frequente.
                             # Un valore più alto (es. 50) rende l'update più stabile. 32 è un buon compromesso.
        batch_size=16,       # Suddividi i 32 campioni in mini-batch da 16 per l'update.
        n_epochs=10,         # Ripeti l'update sui dati raccolti per 10 volte per "spremerli" al massimo.
        verbose=1,
        gamma=0.99           # Gamma è meno importante qui, ma lo manteniamo.
    )

    # 3. Avvia l'apprendimento
    # total_timesteps è ora il NUMERO TOTALE DI SIMULAZIONI che vuoi eseguire.
    # Iniziamo con 250. Se una simulazione dura 2 minuti, il training durerà circa 8.3 ore.
    TOTAL_SIMULATIONS = 250
    print(f"\nAVVIO TRAINING RL per {TOTAL_SIMULATIONS} simulazioni.")
    
    model.learn(total_timesteps=TOTAL_SIMULATIONS, progress_bar=True)

    # 4. Salva il modello finale
    model_path = "ppo_hierfl_optimized"
    model.save(model_path)
    print(f"Training completato. Modello salvato in '{model_path}.zip'")

    # Per vedere la migliore azione trovata, dovresti guardare il tuo file CSV
    # e trovare la riga con lo 'score' più alto.