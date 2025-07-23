import os
import torch
import numpy as np
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import argparse

from OptHierFLEnv import OptHierFLEnv 
from evaluate_model import evaluate_model

def save_conf(filename="conf.txt",conf=None):
    if conf:
        with open(file=filename,mode="w") as f:
            f.write(str(conf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training and evaluation on OptHierFLEnv")
    parser.add_argument("--evaluate", action="store_true",help="Evaluate Model already trained on path output/Experiment_X/ppo_HierFL_best.zip")
    parser.add_argument("--env_name",type=str,default="HierFL-v0", help="name of OptHierFLEnv environment")
    parser.add_argument("--output_dir",type=str,default="output", help="Path to output_dir")
    parser.add_argument("--experiment_id",type=int,default=0,help="Id of the experiment")
    parser.add_argument("--simulate", action="store_true",default=False, help="Use fast simulation instead of real HierFL run")
    


    args = parser.parse_args()
    env_name=args.env_name
    out_dir=args.output_dir
    experiment = args.experiment_id
    

    conf={
    "TOTAL_EPISODES": 400,          # N° totale di episodi completi
    "EPISODES_PER_STEP" : 20,   # N° episodi per ogni iterazione del loop  
    "N_STEPS" : 64,                  # Timesteps per rollout buffer (dev’essere >= EPISODES_PER_STEP*max_rounds)
    "BATCH_SIZE" : 32,                # Min batch_size per training (dev’essere <= N_STEPS)
    "PATIENCE" : 100,                 # Early stopping: episodi senza miglioramento
    "history_len": 3,
    "max_rounds":3,
    "global_rounds":3,
    "epochs":5,
    "random_episodes":5
    }


    # Inizializza l'ambiente
    #max_round-> n° step per episodio 
    env = OptHierFLEnv(env_name=env_name, 
                       max_round=conf["max_rounds"], 
                       global_rounds=conf["global_rounds"] ,
                       history_len=conf["history_len"],
                       simulate=args.simulate)
    
    # Percorsi
    log_path = os.path.join("output", f"Experiment_{experiment}")
    os.makedirs(log_path, exist_ok=True)

    model_path = os.path.join(log_path, "ppo_HierFL")
    best_model_path = os.path.join(log_path, "ppo_HierFL_best")
    results_file = os.path.join(log_path, "training_results.csv")

    #ONLY EVALUATION
    if  args.evaluate:
        model_zip_path = best_model_path + ".zip"
        if os.path.exists(model_zip_path):
            evaluate_model(env=env,weights_path=best_model_path,output_dir=log_path)
        else:
            print(f"ERROR: Model not found at {model_zip_path}")
        exit(0) 

    '''
    total_timesteps (STEP SIZE) >= n_steps (ON PPO )
    Se questo non è rispettato, PPO non esegue alcun aggiornamento della policy, e quindi il training è inutile.
    '''
    
    

    #ogni episodio ha max_round=3 step -> conf["EPISODES_PER_STEP"] * max_round = 6 timesteps per iterazione.
        

    # CSV header
    if not os.path.exists(results_file):
        with open(results_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "AvgReward", "StdDevReward"])

    # Inizializza o carica il modello
    if os.path.exists(model_path + ".zip"):
        print("Carico modello esistente...")
        model = PPO.load(model_path, env=env)
    else:
        print("Creo nuovo modello PPO...")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=conf["N_STEPS"],      # RIDOTTO per HierFL reale
            batch_size=conf["BATCH_SIZE"],  
            n_epochs=5,
            verbose=1
        )

    # Variabili per early stopping
    best_avg_reward = -np.inf
    no_improvement_episodes = 0

    
    print("\nFase di esplorazione iniziale (random policy)...")
    random_episodes = conf["random_episodes"]

    for ep in range(random_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Azione completamente casuale
            obs, reward, done, info = env.step(action)
    
    print(f"Iniziando training con {conf['TOTAL_EPISODES']} episodi totali")
    print(f"Simulazione: {args.simulate}")
    print(f"Parametri PPO: n_steps={conf['N_STEPS']}, batch_size={conf['BATCH_SIZE']}")

    # Training loop basato su episodi
    for episode in range(0, conf["TOTAL_EPISODES"], conf["EPISODES_PER_STEP"]):
        print(f"\n=== Training episodi {episode} - {episode + conf['EPISODES_PER_STEP']} ===")

        # Calcola timesteps necessari per completare gli episodi
        # Ogni episodio ha 3*2 step 
        timesteps_needed = conf["EPISODES_PER_STEP"] * env.max_round 
        
        print(f"Eseguendo {timesteps_needed} timesteps per {conf['EPISODES_PER_STEP']} episodi...")
        
        try:
            # Training
            model.learn(total_timesteps=timesteps_needed, reset_num_timesteps=False)
            
            # Salva modello corrente
            model.save(model_path)
            print(f"Modello salvato: {model_path}")

            # Valutazione
            print("Iniziando valutazione...")
            n_eval_episodes =1  # Meno episodi per HierFL reale
            avg_reward, std_reward = evaluate_policy(
                model, env, 
                n_eval_episodes=5, 
                render=False,
                deterministic=True
            )
            
            current_episode = episode + conf["EPISODES_PER_STEP"]
            print(f"[Valutazione] Episodio {current_episode}: avg_reward = {avg_reward:.4f}, std_dev = {std_reward:.4f}")

            # Scrivi CSV
            with open(results_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_episode, avg_reward, std_reward])

            # Early stopping check
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                no_improvement_episodes = 0
                model.save(best_model_path)
                print(f"Nuovo best model salvato in: {best_model_path} [avg_reward = {avg_reward:.4f}]")
            else:
                no_improvement_episodes += 1
                print(f"Nessun miglioramento ({no_improvement_episodes}/{conf['PATIENCE']})")

            if no_improvement_episodes >= conf["PATIENCE"]:
                print(f"\nEarly stopping attivato dopo {current_episode} episodi. Training interrotto.")
                break
                
        except Exception as e:
            print(f"Errore durante training episodio {episode}: {e}")
            print("Continuando con il prossimo batch...")
            continue

    print("\nTraining completato.")
    print(f"Miglior reward: {best_avg_reward:.4f} | Modello salvato in {best_model_path}")
    save_conf(filename=os.path.join(log_path,"conf.txt"),conf=conf)
