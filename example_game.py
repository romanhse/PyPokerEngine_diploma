from pypokerengine.api.game import setup_config, start_poker
from examples.players.honest_player import HonestPlayer
from examples.players.mlplayerddqn import MLPlayerDDQN
from examples.players.random_player import RandomPlayer
from examples.players.fish_player import FishPlayer

config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
config.register_player(name="ddqn_player", algorithm=MLPlayerDDQN('dueling_ddqn_model_8.pt', 100))
config.register_player(name='fair', algorithm=HonestPlayer())

for i in range(10):
    print(f'Start game {i} from 10 DDQN vs Honest')
    try:
        game_result = start_poker(config, verbose=1)
    except Exception as e:
        print(f'Simulation {i} failed due to {e}')

config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
config.register_player(name="ddqn_player", algorithm=MLPlayerDDQN('dueling_ddqn_model_8.pt', 100))
config.register_player(name='fair', algorithm=RandomPlayer())

for i in range(10):
    print(f'Start game {i} from 10 DDQN vs Random')
    try:
        game_result = start_poker(config, verbose=1)
    except Exception as e:
        print(f'Simulation {i} failed due to {e}')

config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
config.register_player(name="ddqn_player", algorithm=MLPlayerDDQN('dueling_ddqn_model_8.pt', 100))
config.register_player(name='fair', algorithm=FishPlayer())

for i in range(10):
    print(f'Start game {i} from 10 DDQN vs Fish')
    try:
        game_result = start_poker(config, verbose=1)
    except Exception as e:
        print(f'Simulation {i} failed due to {e}')