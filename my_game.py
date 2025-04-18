from pypokerengine.api.game import setup_config, start_poker
from examples.players.fish_player import FishPlayer
from examples.players.console_player import ConsolePlayer
from examples.players.deepseek_player import DeepseekPlayer
from examples.players.emulator_player import EmulatorPlayer
from examples.players.honest_player import HonestPlayer

config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
config.register_player(name="emulator_player", algorithm=HonestPlayer())
config.register_player(name="deepseek_player", algorithm=DeepseekPlayer())
for i in range(10):
    print(f'Start {i}')
    game_result = start_poker(config, verbose=1)  # verbose=0 because game progress is visualized by ConsolePlayer