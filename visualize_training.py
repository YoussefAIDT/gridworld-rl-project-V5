import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import os
import glob


def smooth_data(data, weight=0.9):
    """Lisser les données avec une moyenne mobile exponentielle"""
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def load_tensorboard_data(log_dir):
    """Charger les données depuis TensorBoard"""
    event_files = glob.glob(os.path.join(log_dir, 'PPO_*', 'events.out.tfevents.*'))
    if not event_files:
        event_files = glob.glob(os.path.join(log_dir, 'DQN_*', 'events.out.tfevents.*'))
    
    if not event_files:
        print(f"Aucun fichier TensorBoard trouvé dans {log_dir}")
        return None
    
    event_file = event_files[0]
    print(f"Chargement des données depuis: {event_file}")
    
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Extraire les métriques disponibles
    tags = ea.Tags()['scalars']
    print(f"Métriques disponibles: {tags}")
    
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data


def plot_training_curves(log_dir, save_path='./plots', smooth_weight=0.9):
    """
    Créer des graphiques de visualisation de l'entraînement
    
    Args:
        log_dir: Répertoire des logs TensorBoard
        save_path: Répertoire de sauvegarde des graphiques
        smooth_weight: Poids pour le lissage (0-1)
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Charger les données
    data = load_tensorboard_data(log_dir)
    if data is None:
        return
    
    # Style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Créer une figure avec plusieurs subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Courbes d\'entraînement GridWorld RL', fontsize=16, fontweight='bold')
    
    # 1. Récompense par épisode
    if 'rollout/ep_rew_mean' in data:
        ax = axes[0, 0]
        steps = data['rollout/ep_rew_mean']['steps']
        values = data['rollout/ep_rew_mean']['values']
        smoothed = smooth_data(values, smooth_weight)
        
        ax.plot(steps, values, alpha=0.3, label='Brut', color='blue')
        ax.plot(steps, smoothed, label='Lissé', color='darkblue', linewidth=2)
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Récompense moyenne', fontsize=12)
        ax.set_title('Récompense par épisode', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Longueur d'épisode
    if 'rollout/ep_len_mean' in data:
        ax = axes[0, 1]
        steps = data['rollout/ep_len_mean']['steps']
        values = data['rollout/ep_len_mean']['values']
        smoothed = smooth_data(values, smooth_weight)
        
        ax.plot(steps, values, alpha=0.3, label='Brut', color='green')
        ax.plot(steps, smoothed, label='Lissé', color='darkgreen', linewidth=2)
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Longueur moyenne', fontsize=12)
        ax.set_title('Longueur des épisodes', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Loss (si disponible)
    loss_tags = [tag for tag in data.keys() if 'loss' in tag.lower()]
    if loss_tags:
        ax = axes[1, 0]
        for tag in loss_tags[:3]:  # Max 3 courbes de loss
            steps = data[tag]['steps']
            values = data[tag]['values']
            smoothed = smooth_data(values, smooth_weight)
            ax.plot(steps, smoothed, label=tag.split('/')[-1], linewidth=2)
        
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Fonctions de perte', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Learning rate (si disponible)
    if 'train/learning_rate' in data:
        ax = axes[1, 1]
        steps = data['train/learning_rate']['steps']
        values = data['train/learning_rate']['values']
        
        ax.plot(steps, values, color='red', linewidth=2)
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Taux d\'apprentissage', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé: {plot_path}")
    plt.show()
    
    # Créer un graphique séparé pour l'évolution de la récompense
    if 'rollout/ep_rew_mean' in data:
        plt.figure(figsize=(14, 6))
        steps = data['rollout/ep_rew_mean']['steps']
        values = data['rollout/ep_rew_mean']['values']
        smoothed = smooth_data(values, smooth_weight)
        
        plt.plot(steps, values, alpha=0.2, color='lightblue', label='Données brutes')
        plt.plot(steps, smoothed, color='darkblue', linewidth=2.5, label='Moyenne mobile')
        
        plt.xlabel('Steps d\'entraînement', fontsize=13)
        plt.ylabel('Récompense moyenne par épisode', fontsize=13)
        plt.title('Évolution de la performance de l\'agent', fontsize=15, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        reward_path = os.path.join(save_path, 'reward_evolution.png')
        plt.savefig(reward_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {reward_path}")
        plt.show()


def compare_algorithms(log_dirs, algo_names, save_path='./plots'):
    """
    Comparer plusieurs algorithmes
    
    Args:
        log_dirs: Liste des répertoires de logs
        algo_names: Liste des noms d'algorithmes
        save_path: Répertoire de sauvegarde
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (log_dir, algo_name) in enumerate(zip(log_dirs, algo_names)):
        data = load_tensorboard_data(log_dir)
        if data and 'rollout/ep_rew_mean' in data:
            steps = data['rollout/ep_rew_mean']['steps']
            values = data['rollout/ep_rew_mean']['values']
            smoothed = smooth_data(values, 0.9)
            
            plt.plot(steps, smoothed, label=algo_name, 
                    color=colors[i % len(colors)], linewidth=2.5)
    
    plt.xlabel('Steps d\'entraînement', fontsize=13)
    plt.ylabel('Récompense moyenne', fontsize=13)
    plt.title('Comparaison des algorithmes', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    comparison_path = os.path.join(save_path, 'algorithm_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\nGraphique de comparaison sauvegardé: {comparison_path}")
    plt.show()


def create_performance_summary(results, save_path='./plots'):
    """
    Créer un résumé visuel des performances
    
    Args:
        results: Liste de dictionnaires avec les résultats (ex: issus de evaluate.py)
        save_path: Répertoire de sauvegarde
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Créer un DataFrame
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique 1: Taux de succès
    ax = axes[0]
    sns.barplot(data=df, x='algorithm', y='success_rate', ax=ax, palette='viridis')
    ax.set_ylabel('Taux de succès (%)', fontsize=12)
    ax.set_xlabel('Algorithme', fontsize=12)
    ax.set_title('Taux de succès par algorithme', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    
    # Ajouter les valeurs sur les barres
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    # Graphique 2: Récompense moyenne
    ax = axes[1]
    sns.barplot(data=df, x='algorithm', y='mean_reward', ax=ax, palette='coolwarm')
    ax.set_ylabel('Récompense moyenne', fontsize=12)
    ax.set_xlabel('Algorithme', fontsize=12)
    ax.set_title('Récompense moyenne par algorithme', fontsize=13, fontweight='bold')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    plt.tight_layout()
    summary_path = os.path.join(save_path, 'performance_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"\nRésumé des performances sauvegardé: {summary_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualiser l\'entraînement GridWorld')
    parser.add_argument('--log_dir', type=str, default='./models/logs',
                        help='Répertoire des logs TensorBoard')
    parser.add_argument('--save_path', type=str, default='./plots',
                        help='Répertoire de sauvegarde des graphiques')
    parser.add_argument('--smooth', type=float, default=0.9,
                        help='Poids du lissage (0-1)')
    
    args = parser.parse_args()
    
    # Créer les graphiques
    plot_training_curves(args.log_dir, args.save_path, args.smooth)


if __name__ == '__main__':
    main()
    
    # Exemple d'utilisation:
    # python visualize_training.py --log_dir ./models/logs --save_path ./plots --smooth 0.9