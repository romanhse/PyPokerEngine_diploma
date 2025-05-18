calls = []
raises = []
with open('logs/log_bluff.txt', 'r') as f:
    for line in f:
        line = (line.strip()).split()
        pr = float(line[0][:-1])
        if pr == 0.5:
            continue
        if line[-1] == 'raise,':
            raises.append(pr)
        else:
            calls.append(pr)

print(min(calls))
print(min(raises))
print(sum(calls)/len(calls))
print(sum(raises)/len(raises))
print(len(raises)/len(calls))

import numpy as np
import matplotlib.pyplot as plt

calls = np.array(calls)
raises = np.array(raises)
plt.hist(raises, bins=20, edgecolor='black')  # 20 bins for better resolution
plt.title('Histogram of DDQN raises by hand strength')
plt.xlabel('Strength')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.hist(calls, bins=20, edgecolor='black')  # 20 bins for better resolution
plt.title('Histogram of DDQN calls by hand strength')
plt.xlabel('Strength')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
