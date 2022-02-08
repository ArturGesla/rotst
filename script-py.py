import numpy as np
import matplotlib.pyplot as plt

a=np.loadtxt("res-cd");
plt.plot(a[1:-1]);
plt.savefig("lol.png");
