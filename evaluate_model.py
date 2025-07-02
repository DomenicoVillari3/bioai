import csv
from reinforcement import OptHierFLEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np


def evaluate_agent(agent, env, n_eval_episodes=50, agent_name="PPO"):
    rewards = []
    accuracies = []
    times = []
    energies = []

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = agent.predict(obs) if hasattr(agent, "predict") else (env.action_space.sample(), None)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        accuracies.append(info["raw_results"]["accuracy"])
        times.append(info["raw_results"]["training_time"])
        energies.append(info["raw_results"]["total_energy"])

    return {
        "agent": agent_name,
        "avg_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "avg_accuracy": np.mean(accuracies),
        "avg_time": np.mean(times),
        "avg_energy": np.mean(energies)
    }


def write_results_to_csv(results_list, filename="agent_results.csv"):
    header = ["agent", "avg_reward", "std_reward", "avg_accuracy", "avg_time", "avg_energy"]
    file_exists = False

    try:
        with open(filename, "r"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for result in results_list:
            writer.writerow(result)


if __name__ == "__main__":
    # Inizializza l'ambiente
    env = OptHierFLEnv(env_name="HierFL-v0", max_round=0, global_rounds=3, history_len=1)

    # Carica il modello PPO
    model = PPO.load("ppo_HierFL", env=env)

    # Valutazione
    ppo_results = evaluate_agent(model, env, agent_name="PPO")

    class RandomAgent:
        def predict(self, obs):
            return env.action_space.sample(), None

    random_results = evaluate_agent(RandomAgent(), env, agent_name="Random")

    data = [ppo_results, random_results]

    # Salva su CSV
    write_results_to_csv(data, filename="agent_results.csv")

    # Plot
    labels = ["Avg Reward", "Accuracy", "Training Time", "Energy"]
    ppo_values = [ppo_results["avg_reward"], ppo_results["avg_accuracy"],
                  ppo_results["avg_time"], ppo_results["avg_energy"]]
    rand_values = [random_results["avg_reward"], random_results["avg_accuracy"],
                   random_results["avg_time"], random_results["avg_energy"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ppo_values, width, label='PPO')
    bars2 = ax.bar(x + width/2, rand_values, width, label='Random')

    ax.set_ylabel('Values')
    ax.set_title('PPO vs Random Agent')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
