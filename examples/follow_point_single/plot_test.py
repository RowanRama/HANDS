import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load results
with open("evaluation_outputs.pkl", "rb") as f:
    results = pickle.load(f)

targets = np.array([r["target"] for r in results])
tips = np.array([r["final_tip"] for r in results])
success = np.array([r["success"] for r in results])

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot successful attempts in green
ax.scatter(targets[success, 0], targets[success, 1], targets[success, 2], c='g', label='Target (Success)', marker='o')
ax.scatter(tips[success, 0], tips[success, 1], tips[success, 2], c='lime', label='Tip (Success)', marker='^')

# Plot failed attempts in red
ax.scatter(targets[~success, 0], targets[~success, 1], targets[~success, 2], c='r', label='Target (Fail)', marker='o')
ax.scatter(tips[~success, 0], tips[~success, 1], tips[~success, 2], c='darkred', label='Tip (Fail)', marker='^')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Target vs Final Tip Position (Success = Green, Fail = Red)")
ax.legend()
plt.tight_layout()
plt.show()
