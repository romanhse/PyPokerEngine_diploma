from pypokerengine.api.game import setup_config, start_poker
from examples.players.deepseek_player import DeepseekPlayer
from examples.players.honest_player import HonestPlayer
from examples.players.mlplayerddqn import MLPlayerDDQN

mode = input('Simulate Deepseek vs Honest (type 0), DDQN vs Deepseek (type 1) or Deepseek_New vs Deepseek_old (type 2)')
config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)

if mode == '1':
    config.register_player(name="ddqn_player", algorithm=MLPlayerDDQN('dueling_ddqn_model_7.pt', 100))
    config.register_player(name="deepseek", algorithm=DeepseekPlayer('games_data\deepseek_vs_ddqn.txt', 'games_data\ddqn.txt'))
elif mode == '2':
    config.register_player(name="deepseek_new_player", algorithm=DeepseekPlayer('games_data\deepseek_new.txt', 'games_data\deepseek_old.txt', new_version=True))
    config.register_player(name="deepseek_old_player", algorithm=DeepseekPlayer('games_data\deepseek_old.txt', 'games_data\deepseek_new.txt'))
else:
    config.register_player(name="deepseek", algorithm=DeepseekPlayer('games_data\deepseek.txt', 'games_data\\fair.txt'))
    config.register_player(name='fair', algorithm=HonestPlayer())

for i in range(150):
    print(f'Start {i}')
    try:
        game_result = start_poker(config, verbose=1)
    except Exception as e:
        print(f'Simulation {i} failed due to {e}')
