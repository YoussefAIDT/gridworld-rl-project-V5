import gymnasium as gym
import gridworld_env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import argparse
import os


class TrainingCallback(CheckpointCallback):
    """Callback personnalisé pour logger les informations d'entraînement"""
    def __init__(self, save_freq, save_path, name_prefix, verbose=1):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
        return super()._on_step()


def make_env(env_id, rank=0, seed=0):
    """Créer un environnement avec monitoring"""
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def train_agent(algo_name='ppo', env_id='GridWorld-Simple-v0', 
                total_timesteps=100000, n_envs=4, save_path='./models'):
    """
    Entraîner un agent avec PPO ou DQN
    
    Args:
        algo_name: 'ppo' ou 'dqn'
        env_id: ID de l'environnement
        total_timesteps: Nombre total de steps d'entraînement
        n_envs: Nombre d'environnements parallèles (PPO uniquement)
        save_path: Chemin de sauvegarde
    """
    print(f"\n{'='*60}")
    print(f"Entraînement: {algo_name.upper()} sur {env_id}")
    print(f"{'='*60}\n")
    
    # Créer les dossiers
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/logs", exist_ok=True)
    
    algo_name = algo_name.lower()
    
    if algo_name == 'ppo':
        # Configuration PPO (inspirée de FrozenLake)
        if n_envs > 1:
            env = DummyVecEnv([make_env(env_id, i) for i in range(n_envs)])
        else:
            env = DummyVecEnv([make_env(env_id)])
        
        # Paramètres optimisés pour GridWorld (basés sur FrozenLake)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage l'exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"{save_path}/logs"
        )
        
    elif algo_name == 'dqn':
        # DQN ne supporte pas les environnements vectorisés multiples
        env = gym.make(env_id)
        env = Monitor(env)
        
        # Paramètres optimisés pour GridWorld (basés sur FrozenLake-v1)
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.0001,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=f"{save_path}/logs"
        )
    else:
        raise ValueError(f"Algorithme inconnu: {algo_name}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_path}/checkpoints",
        name_prefix=f"{algo_name}_{env_id}"
    )
    
    # Environnement d'évaluation
    eval_env = gym.make(env_id)
    eval_env = Monitor(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best",
        log_path=f"{save_path}/logs",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Entraînement
    print(f"\nDébut de l'entraînement pour {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Sauvegarder le modèle final
    model_path = f"{save_path}/{algo_name}_{env_id}_final"
    model.save(model_path)
    print(f"\nModèle sauvegardé: {model_path}")
    
    # Évaluation finale
    print("\nÉvaluation finale...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Récompense moyenne: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Fermer les environnements
    env.close()
    eval_env.close()
    
    return model, mean_reward, std_reward


def main():
    parser = argparse.ArgumentParser(description='Entraîner un agent RL sur GridWorld')
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'dqn'],
                        help='Algorithme RL à utiliser')
    parser.add_argument('--env', type=str, default='GridWorld-Simple-v0',
                        choices=['GridWorld-Simple-v0', 'GridWorld-MovingGoals-v0',
                                'GridWorld-MovingObstacles-v0', 'GridWorld-FullDynamic-v0'],
                        help='Environnement GridWorld')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Nombre de timesteps d\'entraînement')
    parser.add_argument('--n_envs', type=int, default=4,
                        help='Nombre d\'environnements parallèles (PPO uniquement)')
    parser.add_argument('--save_path', type=str, default='./models',
                        help='Chemin de sauvegarde des modèles')
    
    args = parser.parse_args()
    
    # Entraîner l'agent
    train_agent(
        algo_name=args.algo,
        env_id=args.env,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
    
    # Exemple d'utilisation:
    # python train.py --algo ppo --env GridWorld-Simple-v0 --timesteps 100000
    # python train.py --algo dqn --env GridWorld-MovingGoals-v0 --timesteps 200000