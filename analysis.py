import os
from matplotlib.cm import get_cmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, t

#
# balances_p1 = \
#     [850, 1270, 1410, 600, 700, 1200, 1100, 760, 1060, 880, 480, 930, 1040, 790, 1000, 1040, 680, 2000, 1220, 770, 900,
#      1250, 540, 800, 660, 490, 970, 670, 790, 1270, 680, 1190, 1130, 1220, 1140, 960, 930, 770, 950, 1180, 500, 1770, 0,
#      2000, 2000, 1670, 1090, 910, 830, 1110, 1130, 1270, 1190, 0, 630, 1160, 1400, 0, 1170, 820, 1380, 1060, 1050, 1300,
#      1160, 1010, 940, 860, 1200, 1290, 1480, 1470, 830, 510, 1130, 1330, 940, 970, 960, 540, 1480, 760, 1270, 820, 1280,
#      1690, 0, 1100, 1300, 280, 1600, 1640, 810, 1190, 1330, 1790, 660, 1180, 1320, 1460]
# balances_p2 = []
# for i in balances_p1:
#     balances_p2.append(2000 - i)
mode = input('Analyze Deepseek vs Honest (type 0), DDQN vs Deepseek (type 1) or Deepseek_New vs Deepseek_old (type 2)')
if mode == '1':
    filenames = ['games_data\ddqn.txt', 'games_data\deepseek_vs_ddqn.txt']
    names = ['DDQN', 'DeepseekDDQN']
    file_pic = 'graphs\ddqn_vs_deepseek'
elif mode == '2':
    filenames = ['games_data\deepseek_new.txt', 'games_data\deepseek_old.txt']
    names = ['Deepseek_New',  'Deepseek_Old']
    file_pic = 'graphs\deepseek_old_vs_deepseek_new'
else:
    filenames = ['games_data\deepseek.txt', 'games_data\\fair.txt']
    names = ['Deepseek', 'Honest']
    file_pic = 'graphs\deepseek_vs_honest'

balances_p1 = []
balances_p2 = []
games_1 = []
games_2 = []
with open(filenames[0], 'r') as f:
    for line in f:
        line = line.rstrip()
        line = list(map(int, line.split()))
        if len(line) == 11 or line[-1] in [5,195,200,0]:
            balances_p1.append(line[-1])
            games_1.append(line)
        else:
            print(line)
with open(filenames[1], 'r') as f:
    for line in f:
        line = line.rstrip()
        line = list(map(int, line.split()))
        if len(line) == 11 or line[-1] in [5,195,200,0]:
            balances_p2.append(line[-1])
            games_2.append(line)
        else:
            print(line)
#Check of correctness:
if len(games_1) != len(games_2):
    raise Exception('Incorrect files to compare - diff # of lines')
for i in range(len(games_1)):
    game_1 = games_1[i]
    game_2 = games_2[i]
    if len(game_1) != len(game_2):
        raise Exception(f'Not equal lines lenghts {game_1}, {game_2}')
    for j in range(len(game_1)):
        if game_1[j] + game_2[j] != 200:
            raise Exception(f'Not summing up to 200 {game_1}, {game_2}')
print(len(games_1))
games = [games_1, games_2]
for i in range(2):
    plt.figure(figsize=(12, 6))

    max_rounds = 11

    for game in games[i]:
        rounds = np.arange(1, len(game) + 1)
        plt.plot(rounds, game, color='blue', alpha=0.2, linewidth=1,
                 label='Отдельные игры' if not plt.gca().lines else "")

    avg_balance = []
    for round_num in range(1, max_rounds + 1):
        round_values = []
        for game in games[i]:
            if len(game) >= round_num:
                round_values.append(game[round_num - 1])
        if round_values:
            avg_balance.append(np.mean(round_values))

    if avg_balance:
        plt.plot(range(1, len(avg_balance) + 1), avg_balance,
                 color='red', linewidth=3, label='Среднее по играм', marker='o')

    plt.title(f'Динамика баланса ({names[i]})')
    plt.xlabel('Номер раунда')
    plt.ylabel('Баланс')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Сохранение
    output_name = f"graphs/{names[i]}_graph.png"
    plt.savefig(output_name)
    plt.close()

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

plt.title(f'Распределение разностей: {names[0]} - {names[1]}')
plt.xlabel('Разность балансов')
plt.ylabel('Частота')
plt.legend()
plt.tight_layout()
plt.savefig(file_pic+'.png', dpi=300)
plt.show()


print(f"Средняя разность: {mean_diff:.2f}")
print(f"95% ДИ (двусторонний): [{ci95_lower:.2f}, {ci95_upper:.2f}]")
print(f"95% ДИ (односторонний): от {ci95:.2f} до ∞")
print(f"t-статистика: {t_stat:.4f}")
print(f"p-значение (двустороннее): {p_value_two_sided}")
print(f"p-значение (одностороннее, проверка: игрок 1 > игрок 2): {p_value_one_sided}")


if t_stat > 0 and p_value_one_sided < 0.05:
    print("✅ Первый игрок статистически значимо лучше (односторонний тест, p < 0.05)")
else:
    print("❌ Нет статистически значимого превосходства (p ≥ 0.05)")

diffs = np.array(balances_p1) - np.array(balances_p2)
observed_mean = np.mean(diffs)

# Bootstrap
n_iterations = 10000
n = len(diffs)
bootstrap_means = []

for _ in range(n_iterations):
    sample = np.random.choice(diffs, size=n, replace=True)
    bootstrap_means.append(np.mean(sample))

bootstrap_means = np.array(bootstrap_means)

# One-sided p-value: proportion of samples ≤ 0
p_value_bootstrap = np.mean(bootstrap_means <= 0)

# Plot
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_means, bins=30, color='lightblue', edgecolor='black')
plt.axvline(observed_mean, color='blue', linestyle='--', label=f'Observed Mean = {observed_mean:.2f}')
plt.axvline(0, color='red', linestyle='-', label='Null Hypothesis (Mean = 0)')
plt.title('Bootstrap Distribution of Mean Differences')
plt.xlabel('Mean Difference')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(file_pic+'_bbotstrap'+'.png', dpi=300)
plt.show()

# Output
print(f"Observed mean difference: {observed_mean:.2f}")
print(f"Bootstrap one-sided p-value ({names[0]} > {names[1]}): {p_value_bootstrap}")