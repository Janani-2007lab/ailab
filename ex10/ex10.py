import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = np.array([
    [.20,.22,.25,.21,.24],
    [.18,.20,.23,.19,.22],
    [.25,.27,.24,.26,.23],
    [.45,.48,.50,.47,.49],
    [.52,.50,.54,.51,.53],
    [.46,.44,.48,.45,.47],
    [.70,.73,.75,.72,.74],
    [.78,.76,.79,.77,.75],
    [.71,.74,.72,.76,.73],
    [.90,.92,.94,.91,.93],
    [.85,.87,.89,.86,.88],
    [.91,.90,.93,.92,.89]
])

y = np.array([0,0,0,1,1,1,2,2,2,3,3,3])

X = np.array([
    [np.mean(x), np.std(x), np.max(x), np.min(x)]
    for x in data
])

a, b, c, d = train_test_split(
    X, y, test_size=4, random_state=42, stratify=y
)

s = StandardScaler()
a = s.fit_transform(a)
b = s.transform(b)

m = KNeighborsClassifier(n_neighbors=3)
m.fit(a, c)

p = m.predict(b)
acc = accuracy_score(d, p)

new = np.array([.72,.74,.73,.75,.71])

f = [[
    np.mean(new),
    np.std(new),
    np.max(new),
    np.min(new)
]]

r = m.predict(s.transform(f))

msg = {
    0: "YES",
    1: "NO",
    2: "HELP",
    3: "WATER"
}

print("NeuroSpeak BCI System")
print("Model Accuracy:", round(acc * 100, 2), "%")
print("New EEG Signal:", new)
print("Predicted Brain Activity:", msg[r[0]])
print("Patient wants to say:", msg[r[0]])