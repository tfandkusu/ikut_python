# %%
import pickle
import pandas as pd

# %%
with open("history_apple.pickle", "rb") as file:
    history_apple = pickle.load(file)
with open("history_nvidia.pickle", "rb") as file:
    history_nvidia = pickle.load(file)
# %%
epochs = []
for i in range(0, 10):
    epochs.append([history_apple["val_accuracy"][i], history_nvidia["val_accuracy"][i]])
# %%
df = pd.DataFrame(
    epochs,
    columns=["apple_val_accuracy", "nvidia_val_accuracy"],
    index=range(1, 11),
)
# %%
ax = df.plot(color=["blue", "green"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
# %%
