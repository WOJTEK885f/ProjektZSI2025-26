import numpy as np
import csv

# Wczytanie testowej bazy
movies = []
with open("movies.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        movies.append(row)

print("Wczytano filmy:")
for m in movies:
    print(" -", m["title"])

# Mapowanie gatunków na liczby
def genre_to_number(genre):
    mapping = {"Action":0, "Drama":1, "Comedy":2}
    return mapping.get(genre, 0)

X = []
for m in movies:
    X.append([
        genre_to_number(m["genre"]),
        float(m["year"]),
        float(m["length"]),
        float(m["rating"])
    ])

X = np.array(X)
X = (X - X.mean(axis=0)) / X.std(axis=0)  # normalizacja

# Losowe "targety", żeby model miał co trenować
y = np.random.rand(len(X), 1)

# Prosta sieć MLP z jedną ukrytą warstwą
np.random.seed(0)

W1 = np.random.randn(4, 5) * 0.1   # 4 cechy - 5 neuronów
b1 = np.zeros((1, 5))

W2 = np.random.randn(5, 1) * 0.1   # 5 neuronów - 1 wynik
b2 = np.zeros((1, 1))


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# Trenowanie
lr = 0.01
epochs = 300

for epoch in range(epochs):
    # forward pass
    z1 = X.dot(W1) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2) + b2
    y_pred = z2

    # loss (MSE)
    loss = np.mean((y_pred - y)**2)

    # backprop
    dloss = 2*(y_pred - y) / len(X)

    dW2 = a1.T.dot(dloss)
    db2 = np.sum(dloss, axis=0, keepdims=True)

    da1 = dloss.dot(W2.T)
    dz1 = da1 * relu_deriv(z1)

    dW1 = X.T.dot(dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 50 == 0:
        print(f"Epoka {epoch}, Loss: {loss:.4f}")

print("\nModel wytrenowany!\n")

# Podnie cech przez użytkownika
print("Podaj swoje preferencje filmu:")

user_genre = input("Gatunek (Action/Drama/Comedy): ")
user_year = float(input("Minimalny rok: "))
user_length = float(input("Minimalna długość: "))
user_rating = float(input("Minimalna ocena IMDb: "))

user_vec = np.array([
    genre_to_number(user_genre),
    user_year,
    user_length,
    user_rating
], dtype=float)

# normalizacja tak samo jak dane
user_vec = (user_vec - X.mean(axis=0)) / X.std(axis=0)

# Obliczamy dopasowanie
print("\nREKOMENDACJE:")

for i, m in enumerate(movies):
    features = X[i]  # już znormalizowane dane filmu

    # połącz cechy użytkownika i filmu
    combined = np.array(features)

    # forward
    z1 = combined.dot(W1) + b1
    a1 = relu(z1)
    score = a1.dot(W2) + b2

    print(f"Film: {m['title']}, Dopasowanie: {float(score):.3f}")