import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.cm import get_cmap

def read_data(filename, num_rounds):
    """Читает данные из файла и разбивает их на игры по указанному количеству раундов."""
    with open(filename, 'r') as f:
        data = [float(line.strip()) for line in f]

    # Автоматически определяем количество игр
    num_games = len(data) // num_rounds
    if len(data) % num_rounds != 0:
        raise ValueError(f"Ошибка: в файле {filename} {len(data)} строк. Это не кратно {num_rounds} раундам на игру")

    # Разделяем данные на игры
    games = [data[i * num_rounds: (i + 1) * num_rounds] for i in range(num_games)]
    return games


def plot_player(games, filename, num_rounds):
    """Строит графики с динамическим количеством раундов"""
    plt.figure(figsize=(12, 6))

    # Находим максимальное количество раундов
    max_rounds = max(len(game) for game in games) if games else 0

    # Рисуем все игры
    for game in games:
        rounds = np.arange(1, len(game) + 1)
        plt.plot(rounds, game, color='blue', alpha=0.2, linewidth=1,
                 label='Отдельные игры' if not plt.gca().lines else "")

    # Рассчитываем среднее с учетом разной длины
    avg_balance = []
    for round_num in range(1, max_rounds + 1):
        round_values = []
        for game in games:
            if len(game) >= round_num:
                round_values.append(game[round_num - 1])
        if round_values:
            avg_balance.append(np.mean(round_values))

    # Рисуем среднее
    if avg_balance:
        plt.plot(range(1, len(avg_balance) + 1), avg_balance,
                 color='red', linewidth=3, label='Среднее по играм', marker='o')

    # Форматирование графика
    base_name = os.path.splitext(os.path.basename(filename))[0]
    plt.title(f'Динамика баланса ({base_name})')
    plt.xlabel('Номер раунда')
    plt.ylabel('Баланс')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Сохранение
    output_name = f"{base_name}_graph.png"
    plt.savefig(output_name)
    plt.close()
    return output_name


def plot_round_evolution(games, filename):
    """Строит график эволюции баланса в раундах между играми"""
    plt.figure(figsize=(14, 8))

    # Определяем параметры данных
    max_rounds = max(len(game) for game in games) if games else 0
    num_games = len(games)

    if max_rounds == 0 or num_games == 0:
        return None

    # Создаем матрицу данных: [раунды][игры]
    round_matrix = []
    for r in range(max_rounds):
        round_values = []
        for g_idx, game in enumerate(games):
            if r < len(game):
                round_values.append(game[r])
            else:
                round_values.append(np.nan)
        round_matrix.append(round_values)

    # Настройка визуализации
    cmap = get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(max_rounds)]
    x_vals = np.arange(1, num_games + 1)

    # Рисуем линии для каждого раунда
    for round_idx in range(max_rounds):
        y_vals = np.array(round_matrix[round_idx])
        valid = ~np.isnan(y_vals)

        if np.any(valid):
            plt.plot(x_vals[valid], y_vals[valid],
                     color=colors[round_idx],
                     marker='o',
                     linestyle='--' if round_idx % 2 else '-',
                     alpha=0.7,
                     label=f'Round {round_idx + 1}')

    # Форматирование графика
    base_name = os.path.splitext(os.path.basename(filename))[0]
    plt.title(f'Эволюция баланса по играм ({base_name})\n'
              f'Каждая линия показывает динамику конкретного раунда')
    plt.xlabel('Номер игры')
    plt.ylabel('Баланс')
    plt.grid(True, alpha=0.3)

    # Интеллектуальное отображение легенды
    if max_rounds <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   title="Раунды:")
    else:
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap),
                     orientation='vertical',
                     label='Номер раунда',
                     boundaries=np.arange(max_rounds + 1))

    # Сохранение
    output_name = f"{base_name}_round_evolution.png"
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()

    return output_name
# Конфигурация (можно менять эти параметры)
NUM_ROUNDS = 11  # Количество раундов в одной игре
PLAYER_FILES = [  # Пути к файлам с данными игроков
    'deepseek.txt',
    'fair.txt'
]

if __name__ == "__main__":
    for player_file in PLAYER_FILES:
        try:
            games = read_data(player_file, NUM_ROUNDS)
            new_game = [float(100)]
            with open('logs.txt', 'r') as f:
                for line in f:
                    if 'won the round' in line:
                        line = line.split()
                        round = int(line[4])
                        deepseek = float(line[-1][:-2])
                        fair = float(line[-3][:-1])
                        if player_file == 'deepseek.txt':
                            if round == 10 or fair == 0 or deepseek == 0:
                                new_game.append(deepseek)
                                games.append(new_game)
                                new_game = [float(100)]
                            else:
                                new_game.append(deepseek)
                        else:
                            if round == 10 or fair == 0 or deepseek == 0:
                                new_game.append(fair)
                                games.append(new_game)
                                new_game = [float(100)]
                            else:
                                new_game.append(fair)
            if player_file == 'deepseek.txt':
                n = 'deepseek'
            else:
                n = 'fair'
            win_g = []
            early_win = 0
            print(f'# of games: {len(games)}')
            for game in games:
                win_g.append(game[-1])
                if game[-1] == float(200):
                    early_win += 1
                if game[0] != float(100):
                    print(game)
            print(f'Avg {n} last round: {sum(win_g) / len(win_g)}')
            print(f'# of early wins of {n} is {early_win}')
            # print(games)
            output = plot_player(games, player_file, NUM_ROUNDS)
            output_evolution = plot_round_evolution(games, player_file)
            print(f"График сохранен как: {output}")

        except FileNotFoundError:
            print(f"Файл {player_file} не найден!")
        except ValueError as e:
            print(f"Ошибка обработки файла {player_file}: {str(e)}")