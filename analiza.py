import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n=== 1. DESCRIEREA SETULUI DE DATE SI A VARIABILEI TINTA ===")
nume_coloane = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'Target']
df = pd.read_csv('australian.dat', sep=r'\s+', names=nume_coloane)
df['Target'] = df['Target'] + 1

numar_instante = df.shape[0]
numar_trasaturi = df.shape[1] - 1 

print(f"Numărul de instanțe/date (rânduri): {numar_instante}")
print(f"Numărul de caracteristici/trasături (trăsături independente): {numar_trasaturi}")


print("\n=== 2. DISTRIBUȚIA CLASELOR ===")
distributie_numar = df['Target'].value_counts().sort_index()
distributie_procent = df['Target'].value_counts(normalize=True).sort_index() * 100

tabel_distributie = pd.DataFrame({
    'Clasa (Target)': distributie_numar.index,
    'Număr instanțe': distributie_numar.values,
    'Procent (%)': distributie_procent.values.round(2)
})
print("Tabel: Număr de instanțe și procent pentru fiecare clasă:")
print(tabel_distributie.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Target', palette='viridis', hue='Target', legend=False)
plt.title('Distribuția Claselor pentru Aprobarea Creditului', fontsize=14)
plt.xlabel('Clasa (1 = Aprobat, 2 = Respins)', fontsize=12)
plt.ylabel('Număr de instanțe', fontsize=12)
plt.show()


print("\n=== 3. IDENTIFICAREA VALORILOR LIPSĂ ȘI DUPLICATE ===")
valori_lipsa = df.isnull().sum()
procent_lipsa = (valori_lipsa / df.shape[0]) * 100
tabel_lipsa = pd.DataFrame({
    'Număr Valori Lipsă': valori_lipsa,
    'Procent (%)': procent_lipsa.round(2)
})
coloane_cu_lipsuri = tabel_lipsa[tabel_lipsa['Număr Valori Lipsă'] > 0]
print("1. Situația valorilor lipsă:")
if coloane_cu_lipsuri.empty:
    print("Nu există valori lipsă în setul de date.")
else:
    print(coloane_cu_lipsuri)
numar_duplicate = df.duplicated().sum()
procent_duplicate = (numar_duplicate / df.shape[0]) * 100

print(f"\n2. Situația instanțelor duplicate:")
print(f"Avem {numar_duplicate} rânduri duplicate, reprezentând {procent_duplicate:.2f}% din date.")

print("\n--- IMPLEMENTARE STRATEGII ---")
if numar_duplicate > 0:
    df = df.drop_duplicates()
    print("-> Rândurile duplicate au fost eliminate.")
else:
    print("-> Nu a fost nevoie de eliminarea duplicatelor.")
if not coloane_cu_lipsuri.empty:
    df = df.fillna(df.median())
    print("-> Valorile lipsă au fost imputate (înlocuite) cu mediana fiecărei coloane.")
print(f"\nDimensiunea setului de date DUPĂ curățare: {df.shape[0]} rânduri, {df.shape[1]} coloane.")


print("\n=== 4. a) ANALIZA TRĂSĂTURILOR NUMERICE ===")
coloane_numerice = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']

statistici = df[coloane_numerice].describe().T
statistici = statistici[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

print("\n1. Tabel cu statistici descriptive:")
print(statistici.round(2).to_string())

print("\n2. Identificare valori posibile de outliers (Metoda IQR):")
for col in coloane_numerice:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    
    if len(outliers) > 0:
        print(f" - {col}: {len(outliers)} outliers (valori < {lower_bound:.1f} sau > {upper_bound:.1f})")
    else:
        print(f" - {col}: 0 outliers")

print("\n3. Se generează histogramele...")

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
fig.suptitle('Histogramele Trăsăturilor Numerice', fontsize=16, fontweight='bold')

for i, col in enumerate(coloane_numerice):
    rand = i // 3
    coloana = i % 3
    sns.histplot(df[col], kde=True, ax=axes[rand, coloana], color='cornflowerblue')
    axes[rand, coloana].set_title(f'Distribuția pentru {col}')
    axes[rand, coloana].set_xlabel('Valoare')
    axes[rand, coloana].set_ylabel('Frecvență')

plt.tight_layout()
plt.show()


print("\n=== 4. b) ANALIZA TRĂSĂTURILOR CATEGORIALE ===")
coloane_categoriale = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']

for col in coloane_categoriale:
    print(f"\n--- Analiza pentru trăsătura: '{col}' ---")
    valori_unice = df[col].value_counts()
    numar_categorii = len(valori_unice)
    proportii = df[col].value_counts(normalize=True) * 100
    tabel_cat = pd.DataFrame({
        'Categorie (Valoare)': valori_unice.index,
        'Număr instanțe': valori_unice.values,
        'Proporție (%)': proportii.values.round(2)
    })
    print(f"Număr total de categorii (cardinalitate): {numar_categorii}")
    print(tabel_cat.to_string(index=False))
    
    categorii_rare = tabel_cat[(tabel_cat['Proporție (%)'] < 5.0) | (tabel_cat['Număr instanțe'] < 10)]
    
    if not categorii_rare.empty:
        print(f"Avertisment: S-au găsit categorii rare!\n{categorii_rare[['Categorie (Valoare)', 'Număr instanțe', 'Proporție (%)']].to_string(index=False)}")
    else:
        print("Nu s-au găsit categorii rare.")
    
    if numar_categorii == 2:
        strategie = "Binară (nu necesită schimbări complexe, poate rămâne 0/1)"
    else:
        strategie = "One-Hot Encoding (Nominală)"
    
    print(f"-> Strategie propusă de codare: {strategie}")


print("\n=== 5. RELAȚIA DINTRE TRĂSĂTURI ȘI TARGET ===")
coloane_corelatie = coloane_numerice + ['Target']
matrice_corelatie = df[coloane_corelatie].corr()

corelatii_target = matrice_corelatie['Target'].drop('Target').sort_values(key=abs, ascending=False)

print("\nCoeficientul de corelație Pearson față de target ('A15'):")
print(corelatii_target.round(3))
plt.figure(figsize=(10, 8))
sns.heatmap(matrice_corelatie, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matricea de Corelație Pearson (Numere vs Target)', fontsize=16)
plt.show()

top_2_trasaturi = corelatii_target.index[:2].tolist()
print(f"\nSe generează graficele pentru cele mai informative 2 trăsături numerice: {top_2_trasaturi}...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Relația dintre cele mai informative 2 trăsături și Target', fontsize=16, fontweight='bold')
for i, col in enumerate(top_2_trasaturi):
    sns.boxplot(x='Target', y=col, data=df, ax=axes[i], palette='Set2', hue='Target', legend=False)
    axes[i].set_title(f'Distribuția {col} în funcție de Target')
    axes[i].set_xlabel('Clasa (1 = Aprobat, 2 = Respins)')
    axes[i].set_ylabel(f'Valoare {col}')

plt.tight_layout()
plt.show()


print("\n=== 6. IMPLEMENTARE RANDOM FOREST ===")
X = df.drop("Target", axis=1)

coloane_nominale = ['A4', 'A5', 'A6', 'A12']
X = pd.get_dummies(X, columns=coloane_nominale, drop_first=False)
print(f"-> S-a aplicat One-Hot Encoding pentru trăsăturile {coloane_nominale}.")
print(f"-> Noua dimensiune a datelor de intrare (X): {X.shape[0]} rânduri, {X.shape[1]} coloane.\n")

y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("-> Datele au fost împărțite în set de antrenare (75%) și testare (25%).")
param_grid = {
    'n_estimators': [10],
    'max_samples': [0.25, 0.40, 0.60, 0.75, 0.90],
    'max_features': ['sqrt', 0.10, 0.50, 0.80, 0.90, 1.0]
}
print("\n-> Se inițializează antrenarea și căutarea celor mai buni hiperparametri (GridSearch)...")
rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("\n-> Căutarea a luat sfârșit!")
print(f"Cea mai buna combinatie de hiperparametri găsită:\n {grid_search.best_params_}")

y_pred = best_rf.predict(X_test)

acuratete = accuracy_score(y_test, y_pred)
print(f"\n================ REZULTATE FINALE ================")
print(f"Acuratete pe setul de test: {acuratete * 100:.2f}%")
print("Raport de Clasificare:\n", classification_report(y_test, y_pred))