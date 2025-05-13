from pypokerengine.api.game import setup_config, start_poker
from examples.players.fish_player import FishPlayer
from examples.players.console_player import ConsolePlayer
from examples.players.deepseek_player import DeepseekPlayer
from examples.players.emulator_player import EmulatorPlayer
from examples.players.honest_player import HonestPlayer
from examples.players.mlplayerddqn import MLPlayerDDQN

config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
config.register_player(name="ddqn_player", algorithm=MLPlayerDDQN('dueling_ddqn_model_7.pt', 100))
config.register_player(name="deepseek_player", algorithm=DeepseekPlayer('deepseek_vs_ddqn.txt', 'ddqn.txt'))
# config.register_player(name = 'nigga', algorithm=HonestPlayer())
# config.register_player(name = 'nigga1', algorithm=HonestPlayer())
res = []
for i in range(100):
    print(f'Start {i}')
    try:
        game_result = start_poker(config, verbose=1)
        for player in game_result["players"]:
            if player["name"] == "ddqn_player":
                print(player["stack"])
                res.append(player["stack"])
    except Exception as e:
        print(f'Simulation {i} failed due to {e}')
print(res)
