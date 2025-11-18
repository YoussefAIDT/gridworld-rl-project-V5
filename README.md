# GridWorld RL avec Stable-Baselines3

Projet d'apprentissage par renforcement sur un environnement GridWorld configurable avec Stable-Baselines3 et RL-Zoo3.

## 📋 Table des matières

- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Environnements disponibles](#environnements-disponibles)
- [Utilisation rapide](#utilisation-rapide)
- [Entraînement](#entraînement)
- [Évaluation](#évaluation)
- [Visualisation](#visualisation)
- [Paramètres des algorithmes](#paramètres-des-algorithmes)
- [Résultats attendus](#résultats-attendus)

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Contenu de requirements.txt

```
gymnasium>=0.29.0
stable-baselines3>=2.0.0
rl-zoo3>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.13.0
pygame>=2.5.0
pandas>=2.0.0
seaborn>=0.12.0
```

## 📁 Structure du projet

```
gridworld_rl/
├── gridworld_env/
│   ├── __init__.py           # Enregistrement des environnements
│   └── gridworld.py          # Environnement GridWorld
├── configs/
│   ├── ppo_gridworld.yml     # Configuration PPO
│   └── dqn_gridworld.yml     # Configuration DQN
├── train.py                  # Script d'entraînement
├── evaluate.py               # Script d'évaluation
├── visualize_training.py     # Script de visualisation
├── run_all.py                # Pipeline complet
├── requirements.txt          # Dépendances
└── README.md                 # Ce fichier
```

## 🎮 Environnements disponibles

### 1. GridWorld-Simple-v0
- Grille 5×5
- Goal fixe en bas à droite
- 3 obstacles fixes
- **Idéal pour débuter**

### 2. GridWorld-MovingGoals-v0
- Grille 8×8
- Goal mobile (probabilité 30%)
- 4 obstacles fixes
- **Difficulté moyenne**

### 3. GridWorld-MovingObstacles-v0
- Grille 10×10
- Goal fixe
- 5 obstacles mobiles (probabilité 20%)
- **Difficulté moyenne-élevée**

### 4. GridWorld-FullDynamic-v0
- Grille 10×10
- 2 goals mobiles
- 6 obstacles mobiles (probabilité 25%)
- **Difficulté maximale**

## ⚡ Utilisation rapide

### Pipeline complet (recommandé)

```bash
# Entraîner PPO sur grille simple + évaluer + visualiser
python run_all.py --algo ppo --env GridWorld-Simple-v0 --all

# Entraîner DQN sur goals mobiles (200k steps)
python run_all.py --algo dqn --env GridWorld-MovingGoals-v0 --timesteps 200000 --all

# Environnement difficile avec PPO (8 envs parallèles)
python run_all.py --algo ppo --env GridWorld-FullDynamic-v0 --timesteps 300000 --n_envs 8 --all
```

## 🎯 Entraînement

### Commande de base

```bash
python train.py --algo ppo --env GridWorld-Simple-v0 --timesteps 100000
```

### Options disponibles

```bash
python train.py \
    --algo ppo \                      # Algorithme: ppo ou dqn
    --env GridWorld-Simple-v0 \       # Environnement
    --timesteps 100000 \              # Nombre de steps
    --n_envs 4 \                      # Envs parallèles (PPO uniquement)
    --save_path ./models              # Dossier de sauvegarde
```

### Exemples d'entraînement

```bash
# PPO sur grille simple (rapide)
python train.py --algo ppo --env GridWorld-Simple-v0 --timesteps 100000

# DQN sur goals mobiles
python train.py --algo dqn --env GridWorld-MovingGoals-v0 --timesteps 200000

# PPO sur environnement complexe (plus long)
python train.py --algo ppo --env GridWorld-FullDynamic-v0 --timesteps 300000 --n_envs 8
```

### Suivre l'entraînement avec TensorBoard

```bash
tensorboard --logdir ./models/logs
```

Ouvrir dans le navigateur: `http://localhost:6006`

## 📊 Évaluation

### Commande de base

```bash
python evaluate.py \
    --model ./models/ppo_GridWorld-Simple-v0_final \
    --algo ppo \
    --env GridWorld-Simple-v0 \
    --episodes 10
```

### Options disponibles

```bash
python evaluate.py \
    --model ./models/ppo_GridWorld-Simple-v0_final \  # Chemin du modèle
    --algo ppo \                                       # Algorithme utilisé
    --env GridWorld-Simple-v0 \                        # Environnement
    --episodes 10 \                                    # Nombre d'épisodes
    --delay 0.3 \                                      # Délai entre steps (sec)
    --no_render                                        # Pas d'affichage console
```

### Exemples d'évaluation

```bash
# Évaluation standard
python evaluate.py --model ./models/ppo_GridWorld-Simple-v0_final --algo ppo --env GridWorld-Simple-v0

# Évaluation détaillée (20 épisodes, ralenti)
python evaluate.py --model ./models/best/best_model --algo ppo --env GridWorld-Simple-v0 --episodes 20 --delay 0.5

# Évaluation rapide sans affichage
python evaluate.py --model ./models/dqn_GridWorld-MovingGoals-v0_final --algo dqn --env GridWorld-MovingGoals-v0 --no_render --episodes 50
```

### Interprétation des résultats

L'évaluation affiche:
- **Taux de succès**: % d'épisodes où le goal est atteint
- **Récompense moyenne**: Performance globale
- **Longueur moyenne**: Nombre de steps par épisode

**Bon résultat**: Taux de succès > 80%, récompense positive, longueur minimale

## 📈 Visualisation

### Créer les graphiques

```bash
python visualize_training.py --log_dir ./models/logs --save_path ./plots
```

### Options disponibles

```bash
python visualize_training.py \
    --log_dir ./models/logs \    # Dossier des logs TensorBoard
    --save_path ./plots \        # Dossier de sortie
    --smooth 0.9                 # Lissage (0-1)
```

### Graphiques générés

1. **training_curves.png**: Vue d'ensemble (récompense, longueur, loss, learning rate)
2. **reward_evolution.png**: Évolution détaillée de la récompense
3. **algorithm_comparison.png**: Comparaison entre algorithmes (si applicable)
4. **performance_summary.png**: Résumé visuel des performances

## 🔧 Paramètres des algorithmes

### PPO (Proximal Policy Optimization)

**Inspirés de FrozenLake-v1 et adaptés à GridWorld**

```python
{
    'learning_rate': 0.0003,
    'n_steps': 2048,         # Steps par environnement avant update
    'batch_size': 64,        # Taille des mini-batchs
    'n_epochs': 10,          # Epochs d'optimisation
    'gamma': 0.99,           # Facteur de discount
    'gae_lambda': 0.95,      # GAE lambda
    'clip_range': 0.2,       # Clip range PPO
    'ent_coef': 0.01,        # Coefficient d'entropie (exploration)
    'vf_coef': 0.5,          # Coefficient value function
    'max_grad_norm': 0.5     # Gradient clipping
}
```

**Pourquoi ces valeurs?**
- `n_steps=2048`: Équilibre entre variance et biais
- `ent_coef=0.01`: Encourage l'exploration dans un espace discret
- `clip_range=0.2`: Valeur standard éprouvée

### DQN (Deep Q-Network)

**Inspirés de FrozenLake-v1 et CartPole-v1**

```python
{
    'learning_rate': 0.0001,
    'buffer_size': 100000,            # Taille du replay buffer
    'learning_starts': 1000,          # Steps avant apprentissage
    'batch_size': 32,                 # Taille des mini-batchs
    'tau': 1.0,                       # Hard update target network
    'gamma': 0.99,                    # Facteur de discount
    'train_freq': 4,                  # Fréquence d'entraînement
    'gradient_steps': 1,              # Steps de gradient par update
    'target_update_interval': 1000,   # Fréquence update target
    'exploration_fraction': 0.1,      # Fraction pour epsilon decay
    'exploration_initial_eps': 1.0,   # Epsilon initial
    'exploration_final_eps': 0.05     # Epsilon final
}
```

**Pourquoi ces valeurs?**
- `buffer_size=100000`: Grande mémoire pour diversité
- `exploration_fraction=0.1`: Exploration rapide puis exploitation
- `target_update_interval=1000`: Stabilité de l'apprentissage

### Ajustements pour environnements dynamiques

Pour **MovingGoals** et **MovingObstacles**:
- Augmenter `ent_coef` (PPO): 0.02-0.03 → Plus d'exploration
- Augmenter `exploration_fraction` (DQN): 0.2-0.25 → Explorer plus longtemps
- Augmenter `n_timesteps`: 200k-300k → Plus de données

## 📈 Résultats attendus

### GridWorld-Simple-v0

| Algo | Timesteps | Taux succès | Récompense moy. | Longueur moy. |
|------|-----------|-------------|-----------------|---------------|
| PPO  | 100k      | 85-95%      | 8-9             | 8-10          |
| DQN  | 100k      | 80-90%      | 7-8             | 9-11          |

### GridWorld-MovingGoals-v0

| Algo | Timesteps | Taux succès | Récompense moy. | Longueur moy. |
|------|-----------|-------------|-----------------|---------------|
| PPO  | 200k      | 70-85%      | 6-8             | 12-16         |
| DQN  | 200k      | 65-80%      | 5-7             | 14-18         |

### GridWorld-FullDynamic-v0

| Algo | Timesteps | Taux succès | Récompense moy. | Longueur moy. |
|------|-----------|-------------|-----------------|---------------|
| PPO  | 300k      | 60-75%      | 4-6             | 18-25         |
| DQN  | 300k      | 55-70%      | 3-5             | 20-28         |

**Note**: Les résultats varient selon le seed et la configuration exacte.

## 🎓 Exemples complets

### Exemple 1: Débutant (Simple)

```bash
# Entraîner PPO pendant 100k steps
python run_all.py --algo ppo --env GridWorld-Simple-v0 --timesteps 100000 --all

# Résultats attendus: ~90% succès en 8-10 steps
```

### Exemple 2: Intermédiaire (Goals mobiles)

```bash
# Entraîner DQN pendant 200k steps
python run_all.py --algo dqn --env GridWorld-MovingGoals-v0 --timesteps 200000 --all

# Résultats attendus: ~75% succès en 14-16 steps
```

### Exemple 3: Avancé (Environnement dynamique)

```bash
# Entraîner PPO avec 8 environnements parallèles
python run_all.py --algo ppo --env GridWorld-FullDynamic-v0 --timesteps 300000 --n_envs 8 --all

# Résultats attendus: ~65% succès en 20-25 steps
```

### Exemple 4: Comparaison d'algorithmes

```bash
# Entraîner PPO
python train.py --algo ppo --env GridWorld-Simple-v0 --timesteps 100000

# Entraîner DQN
python train.py --algo dqn --env GridWorld-Simple-v0 --timesteps 100000

# Évaluer les deux
python evaluate.py --model ./models/ppo_GridWorld-Simple-v0_final --algo ppo --env GridWorld-Simple-v0 --episodes 20
python evaluate.py --model ./models/dqn_GridWorld-Simple-v0_final --algo dqn --env GridWorld-Simple-v0 --episodes 20

# Visualiser et comparer
python visualize_training.py
```

## 🐛 Dépannage

### Problème: "Module 'gridworld_env' not found"

```bash
# Ajouter le dossier au PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

### Problème: Agent ne converge pas

- Augmenter `n_timesteps` (doubler la durée)
- Augmenter `ent_coef` (PPO) ou `exploration_fraction` (DQN)
- Vérifier que l'environnement est solvable (pas trop d'obstacles)

### Problème: Entraînement très lent

- Réduire `n_envs` (PPO) ou `batch_size`
- Utiliser un GPU si disponible
- Réduire la taille de la grille

## 📚 Références

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [RL-Zoo3](https://github.com/DLR-RM/rl-baselines3-zoo)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [DQN Paper](https://arxiv.org/abs/1312.5602)

## 📝 Licence

Projet académique - Libre d'utilisation pour fins éducatives.

---

**Bon apprentissage! 🚀**