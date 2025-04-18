# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def read_data(filename):
#     """Читает данные из файла и разбивает их на игры по 11 раундов."""
#     with open(filename, 'r') as f:
#         data = [float(line.strip()) for line in f]
#     if len(data) != 110:
#         raise ValueError(f"Ошибка: в файле {filename} должно быть 110 строк, найдено {len(data)}")
#     # Разделяем данные на 10 игр по 11 значений
#     games = [data[i * 11: (i + 1) * 11] for i in range(10)]
#     return games
#
#
# def plot_player(games, player_name):
#     """Строит графики для игрока: все игры и среднее значение."""
#     plt.figure(figsize=(12, 6))
#     rounds = np.arange(1, 12)  # Раунды от 1 до 11
#
#     # Рисуем все игры полупрозрачными линиями
#     for i, game in enumerate(games):
#         plt.plot(rounds, game, color='blue', alpha=0.2, linewidth=1,
#                  label='Отдельные игры' if i == 0 else "")
#
#     # Рассчитываем и рисуем среднее по играм
#     avg_balance = np.mean(games, axis=0)
#     plt.plot(rounds, avg_balance, color='red', linewidth=3,
#              label='Среднее по играм', marker='o')
#
#     plt.title(f'Игрок {player_name}: Динамика баланса по раундам')
#     plt.xlabel('Номер раунда')
#     plt.ylabel('Баланс')
#     plt.xticks(rounds)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'player_{player_name}.png')  # Сохраняем график в файл
#     plt.close()
#
#
# # Основная часть программы
# if __name__ == "__main__":
#     players = ['deepseek.txt', 'fair.txt']  # Укажите пути к вашим файлам
#
#     for idx, player_file in enumerate(players, start=1):
#         try:
#             games = read_data(player_file)
#             plot_player(games, str(idx))
#             print(f"График для игрока {idx} сохранен как 'player_{idx}.png'")
#         except FileNotFoundError:
#             print(f"Файл {player_file} не найден!")
#         except ValueError as e:
#             print(e)

import matplotlib.pyplot as plt
import numpy as np
import os


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
    """Строит графики для игрока и сохраняет с именем исходного файла."""
    plt.figure(figsize=(12, 6))
    rounds = np.arange(1, num_rounds + 1)

    # Отдельные игры
    for game in games:
        plt.plot(rounds, game, color='blue', alpha=0.2, linewidth=1,
                 label='Отдельные игры' if not plt.gca().lines else "")

    # Среднее значение
    avg_balance = np.mean(games, axis=0)
    plt.plot(rounds, avg_balance, color='red', linewidth=3,
             label='Среднее по играм', marker='o')

    # Формируем название из имени файла
    base_name = os.path.splitext(os.path.basename(filename))[0]
    plt.title(f'Игрок {base_name}: Динамика баланса по раундам')
    plt.xlabel('Номер раунда')
    plt.ylabel('Баланс')
    plt.xticks(rounds)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Сохраняем график
    output_name = f"{base_name}.png"
    plt.savefig(output_name)
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
            output = plot_player(games, player_file, NUM_ROUNDS)
            print(f"График сохранен как: {output}")

        except FileNotFoundError:
            print(f"Файл {player_file} не найден!")
        except ValueError as e:
            print(f"Ошибка обработки файла {player_file}: {str(e)}")