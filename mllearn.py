from pypokerengine.api.game import setup_config, start_poker
from examples.players.mlplayer import MLPlayer


def train_self_play(n_rounds=1000):
    player_1 = MLPlayer()
    player_2 = MLPlayer()

    config = setup_config(max_round=n_rounds, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="p1", algorithm=player_1)
    config.register_player(name="p2", algorithm=player_2)
    game_result = start_poker(config, verbose=0)

    print("Training done.")


train_self_play()
