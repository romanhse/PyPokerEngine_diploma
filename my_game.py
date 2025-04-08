from pypokerengine.api.game import setup_config, start_poker
from examples.players.fish_player import FishPlayer
from examples.players.console_player import ConsolePlayer
from examples.players.deepseek_player import DeepseekPlayer

config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
# config.register_player(name="fish_player", algorithm=FishPlayer())
config.register_player(name="human_player", algorithm=ConsolePlayer())
config.register_player(name="deepseek_player", algorithm=DeepseekPlayer())
game_result = start_poker(config, verbose=0)  # verbose=0 because game progress is visualized by ConsolePlayer