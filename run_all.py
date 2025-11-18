"""
Script principal pour entraîner, évaluer et visualiser les agents GridWorld
"""

import subprocess
import sys
import os
import argparse


def run_command(command, description):
    """Exécuter une commande avec affichage"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"\n⚠️  Erreur lors de: {description}")
        sys.exit(1)
    print(f"\n✓ {description} - Terminé avec succès")


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline complet d\'entraînement GridWorld RL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

1. Entraîner PPO sur GridWorld simple:
   python run_all.py --algo ppo --env GridWorld-Simple-v0 --timesteps 100000

2. Entraîner DQN sur goals mobiles et visualiser:
   python run_all.py --algo dqn --env GridWorld-MovingGoals-v0 --timesteps 200000 --visualize

3. Tout exécuter (entraîner, évaluer, visualiser):
   python run_all.py --algo ppo --env GridWorld-Simple-v0 --all

4. Seulement évaluer un modèle existant:
   python run_all.py --eval_only --model ./models/ppo_GridWorld-Simple-v0_final --algo ppo --env GridWorld-Simple-v0
        """
    )
    
    # Arguments principaux
    parser.add_argument('--algo', type=str, default='ppo', 
                        choices=['ppo', 'dqn'],
                        help='Algorithme RL (ppo ou dqn)')
    parser.add_argument('--env', type=str, default='GridWorld-Simple-v0',
                        choices=['GridWorld-Simple-v0', 'GridWorld-MovingGoals-v0',
                                'GridWorld-MovingObstacles-v0', 'GridWorld-FullDynamic-v0'],
                        help='Environnement GridWorld')
    
    # Options d'entraînement
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Nombre de timesteps d\'entraînement')
    parser.add_argument('--n_envs', type=int, default=4,
                        help='Nombre d\'environnements parallèles (PPO)')
    
    # Options d'évaluation
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Nombre d\'épisodes d\'évaluation')
    parser.add_argument('--eval_delay', type=float, default=0.3,
                        help='Délai entre steps lors de l\'évaluation')
    
    # Options de workflow
    parser.add_argument('--train_only', action='store_true',
                        help='Seulement entraîner')
    parser.add_argument('--eval_only', action='store_true',
                        help='Seulement évaluer (nécessite --model)')
    parser.add_argument('--model', type=str,
                        help='Chemin vers le modèle (pour eval_only)')
    parser.add_argument('--visualize', action='store_true',
                        help='Créer les graphiques de visualisation')
    parser.add_argument('--all', action='store_true',
                        help='Tout exécuter: entraîner + évaluer + visualiser')
    
    # Chemins
    parser.add_argument('--save_path', type=str, default='./models',
                        help='Chemin de sauvegarde des modèles')
    parser.add_argument('--plot_path', type=str, default='./plots',
                        help='Chemin de sauvegarde des graphiques')
    
    args = parser.parse_args()
    
    # Déterminer le workflow
    if args.all:
        do_train = True
        do_eval = True
        do_viz = True
    elif args.train_only:
        do_train = True
        do_eval = False
        do_viz = False
    elif args.eval_only:
        if not args.model:
            print("❌ Erreur: --model est requis avec --eval_only")
            sys.exit(1)
        do_train = False
        do_eval = True
        do_viz = False
    else:
        # Par défaut: entraîner et évaluer
        do_train = True
        do_eval = True
        do_viz = args.visualize
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║            PIPELINE GRIDWORLD RL - STABLE-BASELINES3              ║
╚═══════════════════════════════════════════════════════════════════╝

Configuration:
  • Algorithme: {args.algo.upper()}
  • Environnement: {args.env}
  • Timesteps: {args.timesteps:,}
  • Épisodes d'évaluation: {args.eval_episodes}

Workflow:
  • Entraînement: {'✓' if do_train else '✗'}
  • Évaluation: {'✓' if do_eval else '✗'}
  • Visualisation: {'✓' if do_viz else '✗'}
    """)
    
    # 1. ENTRAÎNEMENT
    if do_train:
        train_cmd = (
            f"python train.py "
            f"--algo {args.algo} "
            f"--env {args.env} "
            f"--timesteps {args.timesteps} "
            f"--n_envs {args.n_envs} "
            f"--save_path {args.save_path}"
        )
        run_command(train_cmd, f"Entraînement {args.algo.upper()} sur {args.env}")
        
        # Définir le chemin du modèle
        model_path = f"{args.save_path}/{args.algo}_{args.env}_final"
    else:
        model_path = args.model
    
    # 2. ÉVALUATION
    if do_eval:
        eval_cmd = (
            f"python evaluate.py "
            f"--model {model_path} "
            f"--algo {args.algo} "
            f"--env {args.env} "
            f"--episodes {args.eval_episodes} "
            f"--delay {args.eval_delay}"
        )
        run_command(eval_cmd, f"Évaluation du modèle sur {args.eval_episodes} épisodes")
    
    # 3. VISUALISATION
    if do_viz:
        viz_cmd = (
            f"python visualize_training.py "
            f"--log_dir {args.save_path}/logs "
            f"--save_path {args.plot_path} "
            f"--smooth 0.9"
        )
        run_command(viz_cmd, "Création des graphiques de visualisation")
    
    # Résumé final
    print(f"\n{'='*70}")
    print("  ✓ PIPELINE TERMINÉ AVEC SUCCÈS")
    print(f"{'='*70}\n")
    
    if do_train:
        print(f"📁 Modèle sauvegardé: {model_path}")
        print(f"📁 Logs TensorBoard: {args.save_path}/logs")
        print(f"\n💡 Pour visualiser avec TensorBoard:")
        print(f"   tensorboard --logdir {args.save_path}/logs\n")
    
    if do_viz:
        print(f"📊 Graphiques sauvegardés dans: {args.plot_path}/")
    
    print("\n🎮 Commandes utiles:")
    print(f"   • Ré-évaluer: python evaluate.py --model {model_path} --algo {args.algo} --env {args.env}")
    print(f"   • Visualiser: python visualize_training.py --log_dir {args.save_path}/logs")
    print()


if __name__ == '__main__':
    main()


"""
═══════════════════════════════════════════════════════════════════════════
                        EXEMPLES D'UTILISATION
═══════════════════════════════════════════════════════════════════════════

1. ENTRAÎNEMENT COMPLET (PPO sur grille simple):
   python run_all.py --algo ppo --env GridWorld-Simple-v0 --timesteps 100000 --all

2. ENTRAÎNEMENT COMPLET (DQN sur goals mobiles):
   python run_all.py --algo dqn --env GridWorld-MovingGoals-v0 --timesteps 200000 --all

3. ENTRAÎNEMENT LONG (environnement dynamique):
   python run_all.py --algo ppo --env GridWorld-FullDynamic-v0 --timesteps 300000 --n_envs 8 --all

4. SEULEMENT ENTRAÎNER:
   python run_all.py --algo ppo --env GridWorld-Simple-v0 --timesteps 100000 --train_only

5. ÉVALUER UN MODÈLE EXISTANT:
   python run_all.py --eval_only --model ./models/ppo_GridWorld-Simple-v0_final --algo ppo --env GridWorld-Simple-v0 --eval_episodes 20

6. ENTRAÎNER ET VISUALISER (sans évaluation détaillée):
   python run_all.py --algo dqn --env GridWorld-MovingGoals-v0 --train_only --visualize

═══════════════════════════════════════════════════════════════════════════
"""