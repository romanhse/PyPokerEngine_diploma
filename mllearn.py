# from pypokerengine.api.game import setup_config, start_poker
# from examples.players.mlplayer import MLPlayer
#
#
# def train_self_play(n_repeats=1000, n_rounds = 100, stack = 1000, small_blind = 10):
#     player_1 = MLPlayer()
#     player_2 = MLPlayer()
#
#     for i in range(n_repeats):
#         print(f'Game {i}')
#         config = setup_config(max_round=n_rounds, initial_stack=stack, small_blind_amount=small_blind)
#         config.register_player(name="p1", algorithm=player_1)
#         config.register_player(name="p2", algorithm=player_2)
#         game_result = start_poker(config, verbose=0)
#         player_1.save_q_table()
#     print("Training done.")
#
#
# train_self_play()
# import random
# from pypokerengine.api.game import setup_config, start_poker
# from examples.players.mlplayer import MLPlayer
# from examples.players.fish_player import FishPlayer
# from examples.players.random_player import RandomPlayer
# from examples.players.honest_player import HonestPlayer
import matplotlib.pyplot as plt
# import os
#
# def train_self_play_smart(
#         total_games=50,
#         rounds_per_game=100,
#         q_table_path="q_table_1.pkl",
#         show_plot=True
# ):
#     win_history = []
#
#     p1 = MLPlayer()
#     for game_num in range(total_games):
#         print(f"\n🎮 Game {game_num+1}/{total_games}")
#
#         # Choose a random opponent
#         opponent_type = random.choice([HonestPlayer])
#         p2 = opponent_type()
#         print(f'Opponent: {p2}')
#         # if isinstance(p2, MLPlayer):
#         #     p2.load_q_table(q_table_path)
#
#         # Setup game
#         config = setup_config(max_round=rounds_per_game, initial_stack=1000, small_blind_amount=10)
#         config.register_player(name="Trainer", algorithm=p1)
#         config.register_player(name="Opponent", algorithm=p2)
#
#         # Run game
#         result = start_poker(config, verbose=0)
#
#         # Get final stack of trainer
#         for player in result["players"]:
#             if player["name"] == "Trainer":
#                 win_history.append(player["stack"])
#                 print(f"💰 Trainer's stack: {player['stack']}")
#
#
#     print("\n✅ Training Complete.")
#
#     # Plot win history
#     if show_plot:
#         plt.plot(win_history)
#         plt.title("Trainer's Stack Over Games")
#         plt.xlabel("Game")
#         plt.ylabel("Stack")
#         plt.grid()
#         plt.show()
#
# train_self_play_smart()
from examples.players.mlplayerddqn import MLPlayerDDQN
from pypokerengine.api.game import setup_config, start_poker
from examples.players.fish_player import FishPlayer
from examples.players.random_player import RandomPlayer
from examples.players.honest_player import HonestPlayer
from examples.players.all_in_player import AllInPlayer

import random
def train_ddqn_selfplay(n_games=2000, rounds_per_game=100):
    results = []
    for i in range(n_games):
        print(f"🎮 Game {i+1}/{n_games}")

        # p1 = MLPlayerDDQN('dueling_ddqn_model_5.pt')
        p1 = MLPlayerDDQN('dueling_ddqn_model_8.pt', 1000)
        # p2 = MLPlayerDDQN('dueling_ddqn_model_6000.pt')
        # p2 = HonestPlayer()
        opponent_type = random.choice([FishPlayer(), MLPlayerDDQN('dueling_ddqn_model_8.pt', 1000), AllInPlayer(), RandomPlayer()])
        # p2 = FishPlayer()
        # p2 = RandomPlayer()
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
        # for player in game_result["players"]:
        #     if player["name"] == "P1":
        #         results.append(player["stack"])
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
