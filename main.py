import os
import torch
import numpy as np
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import sys

from reinforcement import OptHierFLEnv 
from evaluate_model import evaluate_model




if __name__ == "__main__":
    EXPERIMENT_ID = 2

    # Inizializza l'ambiente
    env = OptHierFLEnv(env_name="HierFL-v0", max_round=3, global_rounds=3, history_len=3)
    
    # Percorsi
    log_path = os.path.join("output", f"Experiment_{EXPERIMENT_ID}")
    os.makedirs(log_path, exist_ok=True)

    model_path = os.path.join(log_path, "ppo_HierFL")
    best_model_path = os.path.join(log_path, "ppo_HierFL_best")
    results_file = os.path.join(log_path, "training_results.csv")

    #ONLY EVALUATION
    if  len(sys.argv)>1 and (sys.argv[1]=="-e" or sys.argv[1]=="--evaluate"):
        if os.path.exists(best_model_path):
            evaluate_model(env=env,weights_path=best_model_path,output_dir=log_path)
        else:
            print("ERROR: NESSUN MODELLO DA IMPORTARE!")
        exit(1)

    
    # Parametri generali
    TOTAL_SIMULATIONS = 750
    STEP_SIZE = 10
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
            learning_rate=3e-4, 
            n_steps=512, #maggior numero di step prima di update
            ent_coef=0.01, #incentiva esplorazione
            batch_size=64,
            n_epochs=10,
            verbose=1,
            gamma=0.99
        )

    # Variabili per early stopping
    best_avg_reward = -np.inf
    no_improvement_steps = 0

    # Training loop a step
    for step in range(0, TOTAL_SIMULATIONS, STEP_SIZE):
        print(f"\nTraining da step {step} a {step + STEP_SIZE}...")

        model.learn(total_timesteps=STEP_SIZE, reset_num_timesteps=False, progress_bar=True)

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
