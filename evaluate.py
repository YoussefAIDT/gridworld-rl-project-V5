import gymnasium as gym
import gridworld_env
from stable_baselines3 import PPO, DQN
import numpy as np
import argparse
import time


def evaluate_agent(model_path, algo_name='ppo', env_id='GridWorld-Simple-v0',
                   n_episodes=10, render=True, delay=0.3):
    """
    Évaluer un agent entraîné
    
    Args:
        model_path: Chemin vers le modèle sauvegardé
        algo_name: 'ppo' ou 'dqn'
        env_id: ID de l'environnement
        n_episodes: Nombre d'épisodes d'évaluation
        render: Afficher ou non la grille
        delay: Délai entre les steps (en secondes)
    """
    print(f"\n{'='*60}")
    print(f"Évaluation: {algo_name.upper()} sur {env_id}")
    print(f"Modèle: {model_path}")
    print(f"{'='*60}\n")
    
    # Charger le modèle
    if algo_name.lower() == 'ppo':
        model = PPO.load(model_path)
    elif algo_name.lower() == 'dqn':
        model = DQN.load(model_path)
    else:
        raise ValueError(f"Algorithme inconnu: {algo_name}")
    
    # Créer l'environnement
    env = gym.make(env_id)
    
    # Statistiques
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"Lancement de {n_episodes} épisodes...\n")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        if render:
            print(f"\n{'='*60}")
            print(f"ÉPISODE {episode + 1}/{n_episodes}")
            print(f"{'='*60}")
            env.render()
        
        while not (terminated or truncated):
            # Prédire l'action
            action, _states = model.predict(obs, deterministic=True)
            
            # Exécuter l'action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render:
                action_names = ['↑ HAUT', '→ DROITE', '↓ BAS', '← GAUCHE']
                print(f"\nAction: {action_names[action]} | Récompense: {reward:.2f}")
                env.render()
                time.sleep(delay)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Vérifier le succès (atteint le goal)
        if episode_reward > 0:  # Récompense positive = goal atteint
            success_count += 1
        
        if render:
            status = "✓ SUCCÈS" if episode_reward > 0 else "✗ ÉCHEC"
            print(f"\n{status} - Récompense totale: {episode_reward:.2f} | Longueur: {episode_length}")
    
    # Afficher les statistiques
    print(f"\n{'='*60}")
    print("STATISTIQUES D'ÉVALUATION")
    print(f"{'='*60}")
    print(f"Nombre d'épisodes: {n_episodes}")
    print(f"Taux de succès: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"\nRécompenses:")
    print(f"  Moyenne: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}")
    print(f"  Max: {np.max(episode_rewards):.2f}")
    print(f"\nLongueurs d'épisode:")
    print(f"  Moyenne: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Min: {np.min(episode_lengths)}")
    print(f"  Max: {np.max(episode_lengths)}")
    print(f"{'='*60}\n")
    
    env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / n_episodes
    }


def main():
    parser = argparse.ArgumentParser(description='Évaluer un agent RL sur GridWorld')
    parser.add_argument('--model', type=str, required=True,
                        help='Chemin vers le modèle à évaluer')
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'dqn'],
                        help='Algorithme RL utilisé')
    parser.add_argument('--env', type=str, default='GridWorld-Simple-v0',
                        choices=['GridWorld-Simple-v0', 'GridWorld-MovingGoals-v0',
                                'GridWorld-MovingObstacles-v0', 'GridWorld-FullDynamic-v0'],
                        help='Environnement GridWorld')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Nombre d\'épisodes d\'évaluation')
    parser.add_argument('--no_render', action='store_true',
                        help='Désactiver l\'affichage')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Délai entre les steps (secondes)')
    
    args = parser.parse_args()
    
    # Évaluer l'agent
    evaluate_agent(
        model_path=args.model,
        algo_name=args.algo,
        env_id=args.env,
        n_episodes=args.episodes,
        render=not args.no_render,
        delay=args.delay
    )


if __name__ == '__main__':
    main()
    
    # Exemples d'utilisation:
    # python evaluate.py --model ./models/ppo_GridWorld-Simple-v0_final --algo ppo --env GridWorld-Simple-v0 --episodes 5
    # python evaluate.py --model ./models/best/best_model --algo dqn --env GridWorld-MovingGoals-v0 --episodes 10 --delay 0.5