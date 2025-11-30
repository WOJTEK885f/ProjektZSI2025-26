import numpy as np
import csv

# ==========================================
# 1. KONFIGURACJA WAG PRZEZ UŻYTKOWNIKA
# ==========================================
print("\n--- KONFIGURACJA WAG (Priorytety) ---")
print("Określ, co jest dla Ciebie ważne w skali 1-3:")
print("(1 - mało ważne, 2 - średnio, 3 - kluczowe)")

try:
    w_genre_usr = float(input(" -> Gatunek (np. Dramat vs Akcja): "))
    w_year_usr = float(input(" -> Rok produkcji: "))
    w_len_usr  = float(input(" -> Długość filmu: "))
    w_budget_usr = float(input(" -> Budżet (Blockbuster vs Kino niezależne): "))
    w_rating_usr = float(input(" -> Ocena (Czy musi być wybitny?): "))
except ValueError:
    print("Błąd! Ustawiam domyślne wagi.")
    w_genre_usr, w_year_usr, w_len_usr, w_budget_usr, w_rating_usr = 3, 1, 1, 1, 2

# Przeliczamy skalę 1-3 na mnożniki do algorytmu (żeby różnice były wyraźniejsze)
W_GENRE  = w_genre_usr * 2.5  # Gatunek wciąż musi być silny
W_YEAR   = w_year_usr
W_LENGTH = w_len_usr * 0.5    # Długość ma mniejszy wpływ naturalnie
W_BUDGET = w_budget_usr
W_RATING = w_rating_usr * 1.5

# ==========================================
# 2. WCZYTANIE DANYCH
# ==========================================
movies = []
filename = "movies.csv"

try:
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies.append(row)
    print(f"\n[INFO] Wczytano {len(movies)} filmów z pliku {filename}.")
except FileNotFoundError:
    print(f"[BŁĄD] Nie znaleziono pliku {filename}. Upewnij się, że jest w folderze!")
    exit()

# ==========================================
# 3. PREPROCESSING (One-Hot + Z-Score)
# ==========================================

# A. One-Hot: Znajdź unikalne gatunki
unique_genres = sorted(list(set(m["genre"] for m in movies)))
print(f"[INFO] Wykryte gatunki: {unique_genres}")

def get_one_hot(genre, all_genres):
    vec = np.zeros(len(all_genres))
    if genre in all_genres:
        vec[all_genres.index(genre)] = 1.0
    return vec

# B. Przygotuj dane do treningu sieci
X_list = []
y_list = [] 

for m in movies:
    g_vec = get_one_hot(m["genre"], unique_genres)
    
    # Numeryczne: Rok, Długość, Budżet
    # Budżet też normalizujemy!
    n_vec = [float(m["year"]), float(m["length"]), float(m["budget"])]
    
    full_vec = np.concatenate((g_vec, n_vec))
    X_list.append(full_vec)
    y_list.append([float(m["rating"])])

X = np.array(X_list)
y = np.array(y_list)

# C. Z-Score dla kolumn numerycznych (ostatnie 3 kolumny: Rok, Długość, Budżet)
# feat_mean i feat_std są KLUCZOWE dla normalizacji użytkownika
num_cols_count = 3 
feat_mean = X[:, -num_cols_count:].mean(axis=0)
feat_std = X[:, -num_cols_count:].std(axis=0) + 1e-8

# Normalizacja danych treningowych
X[:, -num_cols_count:] = (X[:, -num_cols_count:] - feat_mean) / feat_std

# Normalizacja Y (Oceny) do zakresu 0-1
y_max = 10.0
y_norm = y / y_max

# ==========================================
# 4. TRENOWANIE SIECI (MLP)
# ==========================================
print("\n[AI] Trenowanie sieci neuronowej (uczenie się zależności)...")

input_size = X.shape[1]
hidden_size = 12       # Zwiększamy nieco, bo więcej cech
output_size = 1

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

lr = 0.005 # Trochę mniejszy learning rate dla stabilności
epochs = 1500

for epoch in range(epochs):
    # Forward
    z1 = X.dot(W1) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2) + b2
    y_pred = z2
    
    # Backprop
    error = y_pred - y_norm
    loss = np.mean(error**2)
    
    d_loss = 2 * error / len(X)
    dW2 = a1.T.dot(d_loss)
    db2 = np.sum(d_loss, axis=0, keepdims=True)
    da1 = d_loss.dot(W2.T)
    dz1 = da1 * relu_deriv(z1)
    dW1 = X.T.dot(dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

print(f"[AI] Trening zakończony. Loss: {loss:.5f}")

# ==========================================
# 5. DANE OD UŻYTKOWNIKA (PREFERENCJE)
# ==========================================
print("\n=== CZEGO SZUKASZ? ===")
u_genre = input(f"Gatunek ({'/'.join(unique_genres)}): ")
try:
    u_year = float(input("Preferowany rok (np. 2010): "))
    u_length = float(input("Preferowana długość (min): "))
    u_budget = float(input("Preferowany budżet w mln $ (np. 10 - niszowy, 200 - hit): "))
    # Zakładamy, że user szuka filmu idealnego (10/10)
    u_target_rating = 10.0 
except ValueError:
    print("Błąd wprowadzania. Przyjmuję wartości domyślne.")
    u_year, u_length, u_budget = 2000, 120, 50

# Budowa wektora użytkownika
u_vec_genre = get_one_hot(u_genre, unique_genres)
u_vec_nums_raw = np.array([u_year, u_length, u_budget])

# Normalizacja wektora użytkownika TYMI SAMYMI parametrami co bazy
u_vec_nums = (u_vec_nums_raw - feat_mean) / feat_std

# ==========================================
# 6. REKOMENDACJA I WYNIKI
# ==========================================
results = []

for i, m in enumerate(movies):
    movie_vec = X[i] # Znormalizowany wektor filmu z bazy
    
    # Rozbijamy wektor na części
    num_genres = len(unique_genres)
    
    # 1. Część GATUNEK
    m_genre_part = movie_vec[:num_genres]
    dist_genre = np.linalg.norm(m_genre_part - u_vec_genre)
    
    # 2. Część NUMERYCZNA (Rok, Długość, Budżet)
    # Są na końcu wektora
    m_year_norm   = movie_vec[-3]
    m_length_norm = movie_vec[-2]
    m_budget_norm = movie_vec[-1]
    
    # Obliczamy różnice (dystanse) dla każdej cechy
    dist_year   = abs(m_year_norm - u_vec_nums[0])
    dist_length = abs(m_length_norm - u_vec_nums[1])
    dist_budget = abs(m_budget_norm - u_vec_nums[2])
    
    # 3. Część OCENA (chcemy filmy bliskie 10/10)
    # Tu używamy prawdziwej oceny z CSV, nie znormalizowanej
    dist_rating = abs(float(m["rating"]) - u_target_rating)
    
    # --- SUMA WAŻONA ---
    weighted_dist = (dist_genre  * W_GENRE) + \
                    (dist_year   * W_YEAR) + \
                    (dist_length * W_LENGTH) + \
                    (dist_budget * W_BUDGET) + \
                    (dist_rating * W_RATING)
                    
    final_score = 1 / (1 + weighted_dist)

    # Co myśli sieć? (AI Prediction)
    z1 = movie_vec.dot(W1) + b1
    pred = relu(z1).dot(W2) + b2
    ai_rating = pred[0][0] * y_max 

    results.append({
        "title": m["title"],
        "genre": m["genre"],
        "year": m["year"],
        "budget": m["budget"],
        "rating": m["rating"],
        "score": final_score,
        "ai_rating": ai_rating
    })

# Sortowanie malejąco
results.sort(key=lambda x: x["score"], reverse=True)

# Wyświetlanie
print(f"\n{'TYTUŁ':<25} {'GATUNEK':<10} {'ROK':<6} {'BUDŻET':<6} {'OCENA':<6} {'DOPASOWANIE'} {'AI PRED'}")
print("-" * 90)

for r in results[:10]: # Pokaż TOP 10
    bar = "█" * int(r['score'] * 15)
    print(f"{r['title']:<25} {r['genre']:<10} {r['year']:<6} {r['budget']:<6} {r['rating']:<6} {r['score']:.3f} {bar} {r['ai_rating']:.1f}")