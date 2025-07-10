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
    


    # Inizializza l'ambiente
    env = OptHierFLEnv(env_name=env_name, max_round=3, global_rounds=3, history_len=1,simulate=args.simulate)
    
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
    
    # Parametri generali
    # Modalità reale: meno episodi, step più lunghi
    TOTAL_EPISODES = 20
    EPISODES_PER_STEP = 2
    N_STEPS = 8  # Minimo per avere batch_size > 1
    BATCH_SIZE = 4  # Minimo richiesto da PPO
    PATIENCE=5


        

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
            learning_rate=3e-4,
            n_steps=N_STEPS,      # RIDOTTO per HierFL reale
            batch_size=BATCH_SIZE,  # Batch size adattivo
            n_epochs=5,
            gamma=0.99,
            ent_coef=0.01,
            verbose=1
        )

    # Variabili per early stopping
    best_avg_reward = -np.inf
    no_improvement_episodes = 0
    
    print(f"Iniziando training con {TOTAL_EPISODES} episodi totali")
    print(f"Simulazione: {args.simulate}")
    print(f"Parametri PPO: n_steps={N_STEPS}, batch_size={BATCH_SIZE}")

    # Training loop basato su episodi
    for episode in range(0, TOTAL_EPISODES, EPISODES_PER_STEP):
        print(f"\n=== Training episodi {episode} - {episode + EPISODES_PER_STEP} ===")

        # Calcola timesteps necessari per completare gli episodi
        # Ogni episodio ha max_round=3 step
        timesteps_needed = EPISODES_PER_STEP * env.max_round
        
        print(f"Eseguendo {timesteps_needed} timesteps per {EPISODES_PER_STEP} episodi...")
        
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
                n_eval_episodes=8, 
                render=False,
                deterministic=True
            )
            
            current_episode = episode + EPISODES_PER_STEP
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
                print(f"Nessun miglioramento ({no_improvement_episodes}/{PATIENCE})")

            if no_improvement_episodes >= PATIENCE:
                print(f"\nEarly stopping attivato dopo {current_episode} episodi. Training interrotto.")
                break
                
        except Exception as e:
            print(f"Errore durante training episodio {episode}: {e}")
            print("Continuando con il prossimo batch...")
            continue

    print("\nTraining completato.")
    print(f"Miglior reward: {best_avg_reward:.4f} | Modello salvato in {best_model_path}")