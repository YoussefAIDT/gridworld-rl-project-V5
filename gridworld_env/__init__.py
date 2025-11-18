from gymnasium.envs.registration import register
from gridworld_env.gridworld import GridWorldEnv

# Enregistrer différentes configurations
register(
    id='GridWorld-Simple-v0',
    entry_point='gridworld_env.gridworld:GridWorldEnv',
    kwargs={
        'grid_width': 5,
        'grid_height': 5,
        'goal_states': [24],  # Coin bas-droite
        'obstacles': [6, 12, 18],
        'moving_goals': False,
        'moving_obstacles': False
    }
)

register(
    id='GridWorld-MovingGoals-v0',
    entry_point='gridworld_env.gridworld:GridWorldEnv',
    kwargs={
        'grid_width': 8,
        'grid_height': 8,
        'goal_states': [63],
        'obstacles': [9, 18, 27, 36],
        'moving_goals': True,
        'moving_obstacles': False,
        'move_probability': 0.3
    }
)

register(
    id='GridWorld-MovingObstacles-v0',
    entry_point='gridworld_env.gridworld:GridWorldEnv',
    kwargs={
        'grid_width': 10,
        'grid_height': 10,
        'goal_states': [99],
        'obstacles': [15, 25, 35, 45, 55],
        'moving_goals': False,
        'moving_obstacles': True,
        'move_probability': 0.2
    }
)

register(
    id='GridWorld-FullDynamic-v0',
    entry_point='gridworld_env.gridworld:GridWorldEnv',
    kwargs={
        'grid_width': 10,
        'grid_height': 10,
        'goal_states': [99, 89],
        'obstacles': [20, 30, 40, 50, 60, 70],
        'moving_goals': True,
        'moving_obstacles': True,
        'move_probability': 0.25
    }
)
