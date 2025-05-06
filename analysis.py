import os
from matplotlib.cm import get_cmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, t

def read_data(filename):
    games = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            line = list(map(int, line.split()))
            games.append(line)
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
        games = read_data(player_file)
        if player_file == 'deepseek.txt':
            n = 'deepseek'
            balances_p1 = [game[-1] for game in games]
        else:
            n = 'fair'
            balances_p2 = [game[-1] for game in games]
        win_g = []
        early_win = 0
        print(f'# of games: {len(games)}')
        for game in games:
            win_g.append(game[-1])
            if game[-1] == 200:
                early_win += 1
            if game[0] != 100:
                print(game)
        print(f'Avg {n} last round: {sum(win_g) / len(win_g)}')
        print(f'# of early wins of {n} is {early_win}')
        output = plot_player(games, player_file, NUM_ROUNDS)
        output_evolution = plot_round_evolution(games, player_file)
        print(f"График сохранен как: {output}")

    diffs = np.array(balances_p1) - np.array(balances_p2)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    n = len(diffs)
    se = std_diff / np.sqrt(n)
    df = n - 1

    # Двусторонний 95% ДИ
    t_crit_95 = t.ppf(0.975, df)
    ci95_lower = mean_diff - t_crit_95 * se
    ci95_upper = mean_diff + t_crit_95 * se

    # Односторонний 90% ДИ (только нижняя граница)
    t_crit_95 = t.ppf(0.95, df)
    ci95 = mean_diff - t_crit_95 * se

    # Парный t-тест
    t_stat, p_value_two_sided = ttest_rel(balances_p1, balances_p2)
    p_value_one_sided = p_value_two_sided / 2

    # ==== График ====
    plt.figure(figsize=(10, 6))
    sns.histplot(diffs, kde=True, bins=20, color='skyblue')

    # Средняя разность
    plt.axvline(mean_diff, color='blue', linestyle='--', label=f'Средняя разность = {mean_diff:.2f}')

    # Нулевая гипотеза
    plt.axvline(0, color='red', linestyle='-', label='Нулевая гипотеза (0)')

    # Двусторонний ДИ
    plt.axvspan(ci95_lower, ci95_upper, color='blue', alpha=0.2,
                label=f'95% ДИ: [{ci95_lower:.2f}, {ci95_upper:.2f}]')

    # Односторонний ДИ (только нижняя граница)
    plt.axvline(ci95, color='green', linestyle=':', label=f'95% нижн. ДИ: {ci95:.2f}')

    plt.title('Распределение разностей: Deepseek - Honest')
    plt.xlabel('Разность балансов')
    plt.ylabel('Частота')
    plt.legend()
    plt.tight_layout()
    plt.savefig("deepseek_vs_fair.png", dpi=300)
    plt.show()


    print(f"Средняя разность: {mean_diff:.2f}")
    print(f"95% ДИ (двусторонний): [{ci95_lower:.2f}, {ci95_upper:.2f}]")
    print(f"95% ДИ (односторонний): от {ci95:.2f} до ∞")
    print(f"t-статистика: {t_stat:.4f}")
    print(f"p-значение (двустороннее): {p_value_two_sided:.4f}")
    print(f"p-значение (одностороннее, проверка: игрок 1 > игрок 2): {p_value_one_sided:.4f}")


    if t_stat > 0 and p_value_one_sided < 0.05:
        print("✅ Первый игрок статистически значимо лучше (односторонний тест, p < 0.05)")
    else:
        print("❌ Нет статистически значимого превосходства (p ≥ 0.05)")
