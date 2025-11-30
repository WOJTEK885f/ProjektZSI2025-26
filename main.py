import numpy as np
import csv

# --- KONFIGURACJA ---
FILENAME = "movies.csv"

# 1. Wczytanie danych z pliku
movies = []
try:
    with open(FILENAME, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies.append(row)
    print(f"Wczytano {len(movies)} filmów z pliku {FILENAME}.")
except FileNotFoundError:
    print(f"Błąd: Nie znaleziono pliku {FILENAME}. Upewnij się, że plik jest w tym samym folderze.")
    exit()

# 2. Przygotowanie One-Hot Encoding dla gatunków
# Najpierw musimy znaleźć wszystkie unikalne gatunki w bazie
unique_genres = sorted(list(set(m["genre"] for m in movies)))
print(f"Wykryte gatunki: {unique_genres}")

def get_one_hot_vector(genre, all_genres):
    """Zamienia nazwę gatunku na wektor zer i jedynek."""
    vec = np.zeros(len(all_genres))
    if genre in all_genres:
        index = all_genres.index(genre)
        vec[index] = 1.0
    return vec

# 3. Ekstrakcja danych do macierzy NumPy
X_genres = []   # Tu będą wektory One-Hot
X_numerics = [] # Tu będą liczby (Rok, Długość, Ocena)

for m in movies:
    # Część One-Hot
    oh_vec = get_one_hot_vector(m["genre"], unique_genres)
    X_genres.append(oh_vec)
    
    # Część Numeryczna
    nums = [
        float(m["year"]),
        float(m["length"]),
        float(m["rating"])
    ]
    X_numerics.append(nums)

X_genres = np.array(X_genres)
X_numerics = np.array(X_numerics)

# 4. Obliczanie Z-Score dla danych numerycznych (Standaryzacja)
# Wzór: z = (x - mean) / std
numeric_mean = X_numerics.mean(axis=0)
numeric_std = X_numerics.std(axis=0)

# Zabezpieczenie przed dzieleniem przez zero (jeśli np. wszystkie filmy są z tego samego roku)
numeric_std[numeric_std == 0] = 1.0 

X_numerics_norm = (X_numerics - numeric_mean) / numeric_std

# 5. Łączenie cech (One-Hot + Z-Scored Numerics) w jedną macierz
# Każdy film to teraz wektor: [Gatunek_A, Gatunek_B, Gatunek_C, Z_Rok, Z_Długość, Z_Ocena]
X_final = np.hstack((X_genres, X_numerics_norm))

print("\n--- SYSTEM REKOMENDACJI (One-Hot + Z-Score) ---")

# 6. Pobranie danych od użytkownika
print("\nPodaj preferencje:")
u_genre = input(f"Gatunek ({'/'.join(unique_genres)}): ")
try:
    u_year = float(input("Rok: "))
    u_length = float(input("Długość (min): "))
    u_rating = float(input("Ocena (0-10): "))
except ValueError:
    print("Błąd danych. Przyjmuję wartości domyślne.")
    u_year, u_length, u_rating = 2000, 120, 8.0

# 7. Przetworzenie wektora użytkownika TAK SAMO jak danych treningowych

# A. One-Hot dla użytkownika
user_genre_vec = get_one_hot_vector(u_genre, unique_genres)

# B. Z-Score dla liczb użytkownika (używamy średniej i std Z BAZY FILMÓW)
user_numeric_raw = np.array([u_year, u_length, u_rating])
user_numeric_norm = (user_numeric_raw - numeric_mean) / numeric_std

# C. Złączenie w jeden wektor
user_vec_final = np.hstack((user_genre_vec, user_numeric_norm))

# 8. Obliczanie odległości i ranking
results = []

for i, m in enumerate(movies):
    movie_vec = X_final[i]
    
    # Odległość Euklidesowa
    dist = np.linalg.norm(movie_vec - user_vec_final)
    
    # Score (im mniejszy dystans, tym lepiej)
    score = 1 / (1 + dist)
    
    results.append({
        "title": m["title"],
        "genre": m["genre"],
        "year": m["year"],
        "score": score,
        "dist": dist
    })

# 9. Sortowanie malejąco
results.sort(key=lambda x: x["score"], reverse=True)

# 10. Wyświetlanie
print("\n--- WYNIKI ---")
print(f"{'TYTUŁ':<25} {'GATUNEK':<10} {'ROK':<6} {'DOPASOWANIE'}")
print("-" * 60)

for res in results:
    bar = "█" * int(res['score'] * 20) # Dłuższy pasek
    print(f"{res['title']:<25} {res['genre']:<10} {res['year']:<6} {res['score']:.3f} {bar}")