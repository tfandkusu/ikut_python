import pickle

with open("history.pickle", "rb") as file:
    history = pickle.load(file)

val_accuracy = history["val_accuracy"]
print(val_accuracy)
