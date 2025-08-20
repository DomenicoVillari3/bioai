import os
import torch
import numpy as np
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import argparse
import random
import csv

from OptHierFLEnv import OptHierFLEnv 
from evaluate_model import evaluate_model

def save_conf(filename="conf.txt", conf=None):
    """Save configuration to file"""
    if conf:
        with open(file=filename, mode="w") as f:
            f.write(str(conf))

def create_env(env_name, conf, simulate=False):
    """Create and return environment instance"""
    return OptHierFLEnv(
        env_name=env_name, 
        max_round=conf["max_rounds"], 
        global_rounds=conf["global_rounds"],
        history_len=conf["history_len"],
        simulate=simulate
    )

def setup_directories(experiment_id):
    """Create and return necessary directories"""
    log_path = os.path.join("output", f"Experiment_{experiment_id}")
    os.makedirs(log_path, exist_ok=True)
    
    model_path = os.path.join(log_path, "ppo_HierFL")
    best_model_path = os.path.join(log_path, "ppo_HierFL_best")
    results_file = os.path.join(log_path, "training_results.csv")
    test_file=os.path.join(log_path,"res_test.csv")
    
    return log_path, model_path, best_model_path, results_file,test_file

def create_or_load_model(model_path, env, conf, retrain=False):
    """Create new model or load existing one"""
    model_file = model_path + ".zip"
    
    if retrain and os.path.exists(model_file):
        print(f"Loading existing model for retraining: {model_file}")
        # Backup existing model
        backup_path = model_path + "_backup.zip"
        os.system(f'cp {model_file} {backup_path}')
        model = PPO.load(model_path, env=env, tensorboard_log="./ppo_hierfl_tensorboard/")
    elif os.path.exists(model_file) and not retrain:
        print(f"Loading existing model: {model_file}")
        model = PPO.load(model_path, env=env)
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=conf["learning_rate"],
            n_steps=conf["N_STEPS"],
            batch_size=conf["BATCH_SIZE"],  
            n_epochs=conf["epochs"],
            verbose=1,
            ent_coef=conf["ent_coef"])
    
    return model

def random_exploration(env, episodes):
    """Perform random exploration phase"""
    print(f"\nStarting random exploration phase ({episodes} episodes)...")
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
    print("Random exploration completed.")

def test_model(env, model_path, n_episodes=5):
    """Test trained model"""
    model_file = model_path + ".zip"
    if not os.path.exists(model_file):
        print(f"ERROR: Model not found at {model_file}")
        return
    
    print(f"Loading model for testing: {model_file}")
    model = PPO.load(model_path, env=env)
    
    print(f"Testing model for {n_episodes} episodes...")
    obs = env.reset()
    episode_count = 0
    total_reward = 0
    episode_reward = 0
    
    for step in range(500):  # Number of Max steps
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        '''try:
            env.render()
        except Exception as e:
            print(f"Render error: {e}")'''
        
        if done:
            episode_count += 1
            total_reward += episode_reward
            print(f"Episode {episode_count} completed with reward: {episode_reward:.4f}")
            episode_reward = 0
            obs = env.reset()
            
            if episode_count >= n_episodes:
                break
    
    avg_reward = total_reward / episode_count if episode_count > 0 else 0
    print(f"Testing completed. Average reward: {avg_reward:.4f}")
    
    return avg_reward,total_reward,episode_count

def train_model(env, model, conf, results_file, model_path, best_model_path):
    """Training loop"""
    # Initialize CSV
    if not os.path.exists(results_file):
        with open(results_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "AvgReward", "StdDevReward"])

    # Training variables
    best_avg_reward = -np.inf
    no_improvement_episodes = 0
    
    print(f"\nStarting training with {conf['TOTAL_EPISODES']} total episodes")
    print(f"Simulation: {env.simulate}")
    print(f"PPO parameters: n_steps={conf['N_STEPS']}, batch_size={conf['BATCH_SIZE']}")

    # Training loop
    for episode in range(0, conf["TOTAL_EPISODES"], conf["EPISODES_PER_STEP"]):
        print(f"\n=== Training episodes {episode} - {episode + conf['EPISODES_PER_STEP']} ===")

        timesteps_needed = conf["EPISODES_PER_STEP"] * env.max_round 
        print(f"Executing {timesteps_needed} timesteps for {conf['EPISODES_PER_STEP']} episodes...")
        
        try:
            # Training
            model.learn(total_timesteps=timesteps_needed, reset_num_timesteps=False)
            
            # Save current model
            model.save(model_path)
            print(f"Model saved: {model_path}")

            # Evaluation
            print("Starting evaluation...")
            avg_reward, std_reward = evaluate_policy(
                model, env, 
                n_eval_episodes=5, 
                render=False,
                deterministic=True
            )
            
            current_episode = episode + conf["EPISODES_PER_STEP"]
            print(f"[Evaluation] Episode {current_episode}: avg_reward = {avg_reward:.4f}, std_dev = {std_reward:.4f}")

            # Save results to CSV
            with open(results_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_episode, avg_reward, std_reward])

            # Early stopping check
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                no_improvement_episodes = 0
                model.save(best_model_path)
                print(f"New best model saved: {best_model_path} [avg_reward = {avg_reward:.4f}]")
            else:
                no_improvement_episodes += 1
                print(f"No improvement ({no_improvement_episodes}/{conf['PATIENCE']})")

            if no_improvement_episodes >= conf["PATIENCE"]:
                print(f"\nEarly stopping activated after {current_episode} episodes. Training stopped.")
                break
                
        except Exception as e:
            print(f"Error during training episode {episode}: {e}")
            print("Continuing with next batch...")
            continue

    print(f"\nTraining completed. Best reward: {best_avg_reward:.4f}")
    return best_avg_reward

def main():
    # Environment options (can be extended)
    env_list = ["HierFL-v0", "HierFL-v1", "HierFL-v2"]
    
    parser = argparse.ArgumentParser(description="PPO Training and evaluation on OptHierFLEnv")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model using evaluate_model function")
    parser.add_argument("--retrain", action="store_true", help="Load existing model and retrain")
    parser.add_argument("--env_name", type=str, default="HierFL-v0", 
                       help="Name of OptHierFLEnv environment", choices=env_list)
    parser.add_argument("--output_dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("--experiment_id", type=int, default=0, help="Experiment ID")
    parser.add_argument("--simulate", action="store_true", default=False, 
                       help="Use fast simulation instead of real HierFL run")
    parser.add_argument("--model", type=str, default="ppo_HierFL", help="Model name/path")


    args = parser.parse_args()

    # Configuration dictionary
    conf = {
        "TOTAL_EPISODES": 165,
        "EPISODES_PER_STEP": 15,
        "N_STEPS": 128,
        "BATCH_SIZE": 32,
        "PATIENCE": 10,
        "history_len": 3,
        "max_rounds": 3,
        "global_rounds": 3,
        "epochs": 5,
        "random_episodes": 5,
        "ent_coef": 0.001,
        "learning_rate": 3e-4,
        
    }
    

    # Set seeds once
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Create environment
    env = create_env(args.env_name, conf, args.simulate)
    
    # Setup directories
    log_path, model_path, best_model_path, results_file,test_file = setup_directories(args.experiment_id)

    try:
        if args.evaluate:
            # Evaluation mode using evaluate_model function
            model_zip_path = best_model_path + ".zip"
            if os.path.exists(model_zip_path):
                evaluate_model(env=env, weights_path=best_model_path, output_dir=log_path)
            else:
                print(f"ERROR: Model not found at {model_zip_path}")
        
        elif args.test:
            # Test mode - run trained model
            res=test_model(env, best_model_path if os.path.exists(best_model_path + ".zip") else model_path)
            with open(test_file,"w") as f: 
                csv_writer=csv.writer(f)
                csv_writer.writerow(["AvgReward","TotalReward","EpisodeCount"])
                csv_writer.writerow(list(res))
                f.close()
                
        
        else:
            # Training mode
            model = create_or_load_model(model_path, env, conf, args.retrain)
            
            # Random exploration phase
            if not args.retrain:
                random_exploration(env, conf["random_episodes"])
            
            # Training
            best_reward = train_model(env, model, conf, results_file, model_path, best_model_path)
            
            # Save configuration
            save_conf(filename=os.path.join(log_path, "conf.txt"), conf=conf)
            
            print(f"Training completed. Best model saved at: {best_model_path}")
            print(f"Best average reward: {best_reward:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
