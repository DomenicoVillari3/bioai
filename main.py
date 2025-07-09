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
    env = OptHierFLEnv(env_name=env_name, max_round=3, global_rounds=3, history_len=3,simulate=args.simulate)
    
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
    TOTAL_SIMULATIONS = 1024
    STEP_SIZE = 32
    PATIENCE = 5  # Early stopping patience

        

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
        learning_rate=3e-4,    # default SB3, stabile
        n_steps=32,          # Esegui 32 simulazioni prima di un update della policy.
        batch_size=16,       # Suddividi i 32 campioni in mini-batch da 16 per l'update.
        n_epochs=5,         # Ripeti l'update sui dati raccolti per 10 volte
        gamma=0.99
        )

    # Variabili per early stopping
    best_avg_reward = -np.inf
    no_improvement_steps = 0

    # Training loop a step
    for step in range(0, TOTAL_SIMULATIONS, STEP_SIZE):
        print(f"\nTraining da step {step} a {step + STEP_SIZE}...")

        model.learn(total_timesteps=STEP_SIZE, reset_num_timesteps=False)

        # Salva modello corrente
        model.save(model_path)
        print("Modello salvato:", model_path)

        # Valutazione
        avg_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
        print(f"[Valutazione] Step {step + STEP_SIZE}: avg_reward = {avg_reward:.2f}, std_dev = {std_reward:.2f}")

        # Scrivi CSV
        with open(results_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step + STEP_SIZE, avg_reward, std_reward])

        # Early stopping check
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            no_improvement_steps = 0
            model.save(best_model_path)
            print(f"Nuovo best model salvato in: {best_model_path} [avg_reward = {avg_reward:.2f}]")
        else:
            no_improvement_steps += 1
            print(f"Nessun miglioramento ({no_improvement_steps}/{PATIENCE})")

        if no_improvement_steps >= PATIENCE:
            print(f"\nEarly stopping attivato dopo {step + STEP_SIZE} timesteps. Training interrotto.")
            break

    print("\nTraining completato.")
    print(f"Miglior reward: {best_avg_reward:.2f} | Modello salvato in {best_model_path}")
