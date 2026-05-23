import pickle
with open("models/feature_names.pkl", "rb") as f:
    names = pickle.load(f)
print(names)