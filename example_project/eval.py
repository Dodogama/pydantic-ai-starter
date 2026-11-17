import json

from data import make_data
from train import train

(Xtr, ytr), (Xva, yva) = make_data()
m, _ = train(200, Xtr, ytr)
acc = sum(p == t for p, t in zip(m.predict(Xva), yva)) / len(yva)
print(json.dumps({"val_acc": acc}))
