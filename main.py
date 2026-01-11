import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import csv
import threading

# --- KONFIGURACJA WYGLĄDU ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class MovieApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Konfiguracja okna
        self.title("AI Movie Recommender")
        self.geometry("980x800") # Lekko poszerzyłem okno, żeby tabelka ładnie weszła
        self.resizable(True, True)

        # Zmienne modelu
        self.movies = []
        self.unique_genres = []
        self.X = None
        self.feat_mean = None
        self.feat_std = None
        
        # Zmienne sieci neuronowej
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        self.y_max = 10.0
        self.is_trained = False

        # --- GUI LAYOUT ---
        
        # 1. Tytuł
        self.header = ctk.CTkLabel(self, text="🎬 AI Movie Recommender", font=("Roboto", 24, "bold"))
        self.header.pack(pady=10)

        # Główny kontener
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # --- LEWA KOLUMNA: WAGI (Inputy tekstowe zamiast suwaków) ---
        self.left_frame = ctk.CTkFrame(self.main_frame, fg_color="#2b2b2b", corner_radius=10)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(self.left_frame, text="1. KONFIGURACJA WAG (1-3)", font=("Roboto", 16, "bold")).pack(pady=10)
        ctk.CTkLabel(self.left_frame, text="(1 - mało ważne, 3 - kluczowe)", text_color="gray").pack()
        
        # Pola do wpisywania wag
        self.entry_w_genre = self.create_weight_input(self.left_frame, "Waga: Gatunek", "3")
        self.entry_w_year = self.create_weight_input(self.left_frame, "Waga: Rok", "1")
        self.entry_w_len = self.create_weight_input(self.left_frame, "Waga: Długość", "1")
        self.entry_w_budget = self.create_weight_input(self.left_frame, "Waga: Budżet", "1")
        self.entry_w_rating = self.create_weight_input(self.left_frame, "Waga: Ocena", "2")

        # --- PRAWA KOLUMNA: INPUTY PREFERENCJI ---
        self.right_frame = ctk.CTkFrame(self.main_frame, fg_color="#2b2b2b", corner_radius=10)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        ctk.CTkLabel(self.right_frame, text="2. TWOJE PREFERENCJE", font=("Roboto", 16, "bold")).pack(pady=10)

        self.create_label(self.right_frame, "Gatunek:")
        self.combo_genre = ctk.CTkComboBox(self.right_frame, values=["Ładowanie..."])
        self.combo_genre.pack(pady=5)

        self.entry_year = self.create_input(self.right_frame, "Rok (np. 2010):", "2000")
        self.entry_len = self.create_input(self.right_frame, "Długość (min):", "120")
        self.entry_budget = self.create_input(self.right_frame, "Budżet (mln $):", "50")

        # --- PRZYCISK START ---
        self.btn_run = ctk.CTkButton(self, text="CZEKAJ, TRWA ŁADOWANIE...", 
                                     font=("Roboto", 16, "bold"), height=50,
                                     command=self.run_logic,
                                     state="disabled") 
        self.btn_run.pack(pady=10, padx=20, fill="x")

        # --- LOGI I WYNIKI ---
        self.results_box = ctk.CTkTextbox(self, height=250, font=("Consolas", 12))
        self.results_box.pack(pady=10, padx=20, fill="both", expand=True)
        
        # --- URUCHOMIENIE LOGIKI ---
        # 1. Najpierw wczytujemy dane w głównym wątku (żeby GUI działało)
        self.load_data_synchronous()
        
        # 2. Potem odpalamy trening w tle (żeby nie ścięło okna)
        threading.Thread(target=self.train_network_background, daemon=True).start()

    # --- FUNKCJE POMOCNICZE GUI ---
    def create_weight_input(self, parent, label, default_val):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(pady=5, fill="x", padx=10)
        
        lbl = ctk.CTkLabel(frame, text=label, anchor="w")
        lbl.pack(side="left", padx=5)
        
        entry = ctk.CTkEntry(frame, width=50, justify="center")
        entry.insert(0, default_val)
        entry.pack(side="right", padx=5)
        return entry

    def create_input(self, parent, label, placeholder):
        self.create_label(parent, label)
        entry = ctk.CTkEntry(parent, placeholder_text=placeholder)
        entry.pack(pady=5)
        return entry

    def create_label(self, parent, text):
        ctk.CTkLabel(parent, text=text, text_color="#ccc").pack(pady=(5,0))

    def log_msg(self, text):
        # Bezpieczne logowanie z wątku
        self.results_box.insert("end", text + "\n")
        self.results_box.see("end")

    def get_one_hot(self, genre):
        vec = np.zeros(len(self.unique_genres))
        if genre in self.unique_genres:
            vec[self.unique_genres.index(genre)] = 1.0
        return vec

    # --- LOGIKA DANYCH (GŁÓWNY WĄTEK) ---
    def load_data_synchronous(self):
        """Wczytuje CSV i ustawia ComboBox. Musi być w głównym wątku."""
        filename = "movies.csv"
        raw_backup = [
            {"title": "The Matrix", "genre": "Action", "year": 1999, "length": 136, "budget": 63, "rating": 8.7},
            {"title": "The Godfather", "genre": "Drama", "year": 1972, "length": 175, "budget": 6, "rating": 9.2},
            {"title": "Inception", "genre": "Action", "year": 2010, "length": 148, "budget": 160, "rating": 8.8},
        ]
        
        try:
            with open(filename, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.movies.append(row)
            self.log_msg(f"[INFO] Wczytano {len(self.movies)} filmów.")
        except FileNotFoundError:
            self.movies = raw_backup
            self.log_msg(f"[INFO] Brak pliku {filename}. Używam danych testowych.")

        # Aktualizacja listy gatunków (TERAZ BEZPIECZNA)
        self.unique_genres = sorted(list(set(m["genre"] for m in self.movies)))
        self.combo_genre.configure(values=self.unique_genres)
        if self.unique_genres: self.combo_genre.set(self.unique_genres[0])

        # Preprocessing danych
        X_list = []
        y_list = []

        for m in self.movies:
            g_vec = self.get_one_hot(m["genre"])
            n_vec = [float(m["year"]), float(m["length"]), float(m["budget"])]
            X_list.append(np.concatenate((g_vec, n_vec)))
            y_list.append([float(m["rating"])])

        self.X = np.array(X_list)
        y = np.array(y_list)

        # Normalizacja
        self.feat_mean = self.X[:, -3:].mean(axis=0)
        self.feat_std = self.X[:, -3:].std(axis=0) + 1e-8
        self.X[:, -3:] = (self.X[:, -3:] - self.feat_mean) / self.feat_std
        
        self.y_norm = y / self.y_max

    # --- TRENING (WĄTEK TŁA) ---
    def train_network_background(self):
        self.log_msg(">>> Start treningu sieci neuronowej w tle...")
        
        input_size = self.X.shape[1]
        hidden_size = 12
        output_size = 1

        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        def relu(x): return np.maximum(0, x)
        def relu_deriv(x): return (x > 0).astype(float)

        lr = 0.005
        epochs = 1500
        
        for epoch in range(epochs):
            z1 = self.X.dot(self.W1) + self.b1
            a1 = relu(z1)
            z2 = a1.dot(self.W2) + self.b2
            y_pred = z2
            
            error = y_pred - self.y_norm
            loss = np.mean(error**2)
            
            d_loss = 2 * error / len(self.X)
            dW2 = a1.T.dot(d_loss)
            db2 = np.sum(d_loss, axis=0, keepdims=True)
            da1 = d_loss.dot(self.W2.T)
            dz1 = da1 * relu_deriv(z1)
            dW1 = self.X.T.dot(dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            self.W1 -= lr * dW1; self.b1 -= lr * db1
            self.W2 -= lr * dW2; self.b2 -= lr * db2

        self.log_msg(f"[AI] Trening zakończony. Loss: {loss:.5f}")
        self.is_trained = True
        
        # Odblokowanie przycisku (musi być z głównego wątku lub bezpiecznie)
        self.btn_run.configure(text="GENERUJ REKOMENDACJE", state="normal")

    # --- LOGIKA SZUKANIA ---
    def run_logic(self):
        # 1. Definicja zmiennej Rating PRZED try/catch (Naprawa NameError)
        u_target_rating = 10.0 

        # 2. Pobranie danych z GUI
        try:
            u_genre = self.combo_genre.get()
            u_year = float(self.entry_year.get())
            u_len = float(self.entry_len.get())
            u_budget = float(self.entry_budget.get())
            
            # Pobranie wag z pól tekstowych
            w_genre = float(self.entry_w_genre.get())
            w_year = float(self.entry_w_year.get())
            w_len = float(self.entry_w_len.get())
            w_budget = float(self.entry_w_budget.get())
            w_rating = float(self.entry_w_rating.get())
            
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawne liczby w polach!")
            return

        # Przeliczenie wag (Logika z Twojego kodu)
        W_GENRE = w_genre * 2.5
        W_YEAR = w_year
        W_LENGTH = w_len * 0.5
        W_BUDGET = w_budget
        W_RATING = w_rating * 1.5

        self.results_box.delete("1.0", "end")
        self.log_msg(f"Szukam: {u_genre} (Waga: {w_genre}), Rok: {u_year}...")

        # Wektor usera
        u_vec_genre = self.get_one_hot(u_genre)
        u_vec_nums_raw = np.array([u_year, u_len, u_budget])
        u_vec_nums = (u_vec_nums_raw - self.feat_mean) / self.feat_std

        results = []
        def relu(x): return np.maximum(0, x)

        for i, m in enumerate(self.movies):
            movie_vec = self.X[i]
            
            # Dystanse
            m_genre_part = movie_vec[:len(self.unique_genres)]
            dist_genre = np.linalg.norm(m_genre_part - u_vec_genre)
            
            dist_year = abs(movie_vec[-3] - u_vec_nums[0])
            dist_len = abs(movie_vec[-2] - u_vec_nums[1])
            dist_budget = abs(movie_vec[-1] - u_vec_nums[2])
            dist_rating = abs(float(m["rating"]) - u_target_rating)

            weighted_dist = (dist_genre * W_GENRE) + \
                            (dist_year * W_YEAR) + \
                            (dist_len * W_LENGTH) + \
                            (dist_budget * W_BUDGET) + \
                            (dist_rating * W_RATING)
            
            final_score = 1 / (1 + weighted_dist)

            # Predykcja sieci
            z1 = movie_vec.dot(self.W1) + self.b1
            pred = relu(z1).dot(self.W2) + self.b2
            ai_rating = pred[0][0] * self.y_max

            results.append((m, final_score, ai_rating))

        results.sort(key=lambda x: x[1], reverse=True)

        header = f"{'TYTUŁ':<25} {'GATUNEK':<10} {'ROK':<6} {'CZAS':<6} {'BUDŻET':<6} {'OCENA':<6} {'SCORE'}   {'AI PRED'}"
        self.log_msg("\n" + header)
        self.log_msg("-" * 105)
        
        for r in results[:10]:
            m, sc, ai = r
            line = f"{m['title'][:23]:<25} {m['genre'][:9]:<10} {m['year']:<6} {m['length']:<6} {m['budget']:<6} {m['rating']:<6} {sc:.3f}   {ai:.1f}"
            self.log_msg(line)

if __name__ == "__main__":
    app = MovieApp()
    app.mainloop()