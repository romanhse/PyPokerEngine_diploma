from examples.players.mlplayerddqn import MLPlayerDDQN
from pypokerengine.api.game import setup_config, start_poker
from examples.players.fish_player import FishPlayer
from examples.players.random_player import RandomPlayer
from examples.players.honest_player import HonestPlayer
from examples.players.all_in_player import AllInPlayer

import random
def train_ddqn_selfplay(n_games=3000, rounds_per_game=100):
    results = []
    for i in range(n_games):
        print(f"🎮 Game {i+1}/{n_games}")
        p1 = MLPlayerDDQN('dueling_ddqn_model_7.pt', 1000)
        opponent_type = random.choice([FishPlayer(), MLPlayerDDQN('dueling_ddqn_model_7.pt', 1000),  RandomPlayer()])
        p2 = opponent_type
        order = random.choice([True, False])
        if order:
            config = setup_config(max_round=rounds_per_game, initial_stack=1000, small_blind_amount=10)
            config.register_player(name="P1", algorithm=p1)
            config.register_player(name="P2", algorithm=p2)
        else:
            config = setup_config(max_round=rounds_per_game, initial_stack=1000, small_blind_amount=10)
            config.register_player(name="P2", algorithm=p2)
            config.register_player(name="P1", algorithm=p1)

        game_result = start_poker(config, verbose=0)
        if order:
            results.append(game_result['players'][0]['stack'])
            print("🧠 Epsilon:", round(p1.epsilon, 3), "| Stack:", game_result['players'][0]['stack'], "Player: " ,p2)
        else:
            results.append(game_result['players'][1]['stack'])
            print("🧠 Epsilon:", round(p1.epsilon, 3), "| Stack:", game_result['players'][1]['stack'], "Player: ", p2)

    print("✅ Training done.")
    print(results)
    print(sum(results)/len(results))
train_ddqn_selfplay()
