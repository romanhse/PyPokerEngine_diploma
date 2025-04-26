# from pypokerengine.api.game import setup_config, start_poker
# from examples.players.mlplayer import MLPlayer
# from examples.players.honest_player import HonestPlayer
#
# def play_trained_vs_fish():
#     player1 = MLPlayer()
#     player1.load_q_table()  # load trained knowledge
#
#     player2 = HonestPlayer()  # dumb rule-based bot
#
#     config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=10)
#     config.register_player(name="TrainedBot", algorithm=player1)
#     config.register_player(name="Honest", algorithm=player2)
#
#     game_result = start_poker(config, verbose=1)
#
#     print("\n🏁 Game Over")
#     print("Stacks:", game_result["players"])
#
# if __name__ == "__main__":
#     play_trained_vs_fish()
import random
from pypokerengine.api.game import setup_config, start_poker
from examples.players.mlplayer import MLPlayer
from examples.players.fish_player import FishPlayer
from examples.players.random_player import RandomPlayer
from examples.players.honest_player import HonestPlayer
import matplotlib.pyplot as plt
import os

def train_self_play_smart(
        total_games=10000,
        rounds_per_game=100,
        q_table_path="q_table_1.pkl",
        prune_limit=1000000,
        show_plot=True
):
    win_history = []

    p1 = MLPlayer()
    for game_num in range(total_games):
        print(f"\n🎮 Game {game_num+1}/{total_games}")

        # Create players
        p1.load_q_table(q_table_path)

        # Choose a random opponent
        opponent_type = random.choice([FishPlayer, HonestPlayer, RandomPlayer, MLPlayer])
        p2 = opponent_type()
        print(f'Opponent: {p2}')
        # if isinstance(p2, MLPlayer):
        #     p2.load_q_table(q_table_path)

        # Setup game
        config = setup_config(max_round=rounds_per_game, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="Trainer", algorithm=p1)
        config.register_player(name="Opponent", algorithm=p2)

        # Run game
        result = start_poker(config, verbose=0)

        # Get final stack of trainer
        for player in result["players"]:
            if player["name"] == "Trainer":
                win_history.append(player["stack"])
                print(f"💰 Trainer's stack: {player['stack']}")

        # Save trained brain
        p1.save_q_table(q_table_path)

        # Epsilon decay
        if hasattr(p1, "epsilon"):
            p1.epsilon = max(0.1, p1.epsilon * 0.995)

        # Optional Q-table cleanup
        if hasattr(p1, "q_table") and len(p1.q_table) > prune_limit:
            print("⚠️ Q-table too big, pruning...")
            p1.q_table = dict(random.sample(p1.q_table.items(), prune_limit // 2))

    print("\n✅ Training Complete.")

    # Plot win history
    if show_plot:
        plt.plot(win_history)
        plt.title("Trainer's Stack Over Games")
        plt.xlabel("Game")
        plt.ylabel("Stack")
        plt.grid()
        plt.show()

train_self_play_smart()