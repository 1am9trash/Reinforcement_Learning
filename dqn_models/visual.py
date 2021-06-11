# use to visualize the score

import matplotlib.pyplot as plt
import numpy as np

x = np.zeros((1001))
y = np.zeros((1001))

for i in range(1001):
    x[i], y[i] = map(float, input().split(','))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Episode")
ax.set_ylabel("Score")
plt.plot(x, y, color='blue', linewidth=0.5)
plt.savefig('result.png')
plt.show()
