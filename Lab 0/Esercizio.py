"""1. Load the provided .csv file with the used car data"""

import numpy as np

file_name = "km_year_power_price.csv"

# Inizializza quattro liste vuote
km = []
year = []
power = []
price = []

# Apre il file
infile = open(file_name, 'r')
line_c = 0
for line in infile:
    if line_c > 0:  # Legge dalla seconda riga (per saltare l'header)
        line = line.strip()  # Rimuove spazi bianchi e newline
        v = line.split(',')  # Divide la riga in una lista

        # Aggiunge gli elementi della lista nei rispettivi vettori
        km.append(int(v[0].strip()))     # Converti il km in intero e aggiungi alla lista km
        year.append(int(v[1].strip()))   # Converti l'anno in intero e aggiungi alla lista year
        power.append(int(v[2].strip()))  # Converti la potenza in intero e aggiungi alla lista power
        price.append(float(v[3].strip())) # Converti il prezzo in float e aggiungi alla lista price

    line_c += 1

# Chiude il file
infile.close()

# Converte le liste in array numpy
km = np.array(km)
year = np.array(year)
power = np.array(power)
price = np.array(price)

"""Use a linear regression to estimate the car prices from the year, kilometers or engine power. You can make a simple 1D regression from each one of the parameters independently (as an optional task you can also try a 2D or 3D regression combining multiple cues)
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dati caricati dai vettori (array) di prima
# Supponiamo di voler usare 'km' come variabile indipendente per prevedere 'price'
X = km.reshape(-1, 1)  # km è la variabile indipendente (reshape per renderlo una matrice, scikit lo vuole così)
y = price  # price è la variabile dipendente (il target)

# Crea il modello di regressione lineare
model = LinearRegression()

# Esegui il training del modello con i dati
model.fit(X, y)

# Ora possiamo fare le predizioni dei prezzi in base ai chilometri
y_pred = model.predict(X)

r2_km = r2_score(price, y_pred)

""" Grafico della regressione
plt.scatter(X, y, color='blue')  # Punti reali (dati originali)
plt.plot(X, y_pred, color='red')  # Linea di regressione
plt.title('Regressione lineare: km vs price')
plt.xlabel('Chilometri (km)')
plt.ylabel('Prezzo medio (avgPrice)')
plt.show() """

#Ora facciamo la stessa cosa con l'anno e la potenza

X = year.reshape(-1, 1)  # km è la variabile indipendente (reshape per renderlo una matrice, scikit lo vuole così)
y = price  # price è la variabile dipendente (il target)

# Crea il modello di regressione lineare
model = LinearRegression()

# Esegui il training del modello con i dati
model.fit(X, y)

# Ora possiamo fare le predizioni dei prezzi in base ai chilometri
y_pred = model.predict(X)

r2_year = r2_score(price, y_pred)

"""# Grafico della regressione
plt.scatter(X, y, color='blue')  # Punti reali (dati originali)
plt.plot(X, y_pred, color='red')  # Linea di regressione
plt.title('Regressione lineare: year vs price')
plt.xlabel('Anni')
plt.ylabel('Prezzo medio (avgPrice)')
plt.show()"""

X = power.reshape(-1, 1)  # km è la variabile indipendente (reshape per renderlo una matrice, scikit lo vuole così)
y = price  # price è la variabile dipendente (il target)

# Crea il modello di regressione lineare
model = LinearRegression()

# Esegui il training del modello con i dati
model.fit(X, y)

# Ora possiamo fare le predizioni dei prezzi in base ai chilometri
y_pred = model.predict(X)

r2_power = r2_score(price, y_pred)

# Stampa i coefficienti della regressione
#print(f"Intercept: {model.intercept_}")
print(f"Coefficiente usando la potenza: {model.coef_[0]}")

"""# Grafico della regressione
plt.scatter(X, y, color='blue')  # Punti reali (dati originali)
plt.plot(X, y_pred, color='red')  # Linea di regressione
plt.title('Regressione lineare: Power vs price')
plt.xlabel('Anni')
plt.ylabel('Prezzo medio (avgPrice)')
plt.show()"""

"""Ora procedo con una regressione 2D"""

# Combina km e year come input (variabili indipendenti)
X_multi = np.column_stack((km, year))

# Crea il modello di regressione lineare
model_multi = LinearRegression()

# Addestra il modello usando km e year per prevedere il prezzo
model_multi.fit(X_multi, price)

# Previsione dei prezzi usando il modello
y_pred_multi = model_multi.predict(X_multi)
r2_km_year = r2_score(price, y_pred_multi)

# Mostra le previsioni rispetto ai dati reali
#for i in range(len(km)):
    #print(f"Chilometri: {km[i]}, Anno: {year[i]}, Prezzo reale: {price[i]}, Prezzo predetto: {y_pred_multi[i]}")

# Facoltativo: puoi anche tracciare i risultati con un grafico (anche se è difficile visualizzare più variabili insieme)
"""plt.scatter(km, price, color='blue', label='Prezzo Reale')
plt.scatter(km, y_pred_multi, color='red', label='Prezzo Predetto')
plt.title('Regressione 2D: km e year vs price')
plt.xlabel('Chilometri (km)')
plt.ylabel('Prezzo medio (avgPrice)')
plt.legend()
plt.show()"""

"""Regressione 3D"""
# Combina km, year e power come input (variabili indipendenti)
X_multi_3D = np.column_stack((km, year, power))

# Crea il modello di regressione lineare
model_multi_3D = LinearRegression()

# Addestra il modello usando km, year e power per prevedere il prezzo
model_multi_3D.fit(X_multi_3D, price)

# Previsione dei prezzi usando il modello
y_pred_multi_3D = model_multi_3D.predict(X_multi_3D)
r2_km_year_power = r2_score(price, y_pred_multi_3D)

"""# Creazione del grafico per confrontare i prezzi reali e predetti
plt.figure(figsize=(10,6))
plt.plot(range(1, len(price) + 1), price, 'o-', label='Prezzo Reale', color='blue', markersize=8)
plt.plot(range(1, len(y_pred_multi_3D) + 1), y_pred_multi_3D, 's--', label='Prezzo Predetto', color='green', markersize=8)

# Aggiunta di dettagli al grafico
plt.title('Confronto tra Prezzo Reale e Prezzo Predetto per Ogni Macchina')
plt.xlabel('Macchina')
plt.ylabel('Prezzo')
plt.legend()
plt.grid(True)
plt.show()"""

print("Usando SciKit-Learn")
print(f"R^2 usando i chilometri: {r2_km:.4f}")
print(f"R^2 usando l'anno: {r2_year:.4f}")
print(f"R^2 usando la potenza: {r2_power:.4f}")
print(f"R^2 usando l'anno e i chilometri: {r2_km_year:.4f}")
print(f"R^2 usando tutte e tre le variabili: {r2_km_year_power:.4f}")


"""Rieseguo le regressioni lineari 1D usando ora scipy linregress"""

from scipy.stats import linregress

# Funzione per calcolare la regressione lineare e creare il grafico
def plot_regression(x, y, xlabel, ylabel, title):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Calcola la retta di regressione
    line = slope * x + intercept
    
    """# Visualizza il grafico
    plt.scatter(x, y, color='blue', label='Dati Reali')
    plt.plot(x, line, color='red', label='Retta di Regressione')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()"""
    
    # Ritorna i parametri della regressione e il valore di R²
    return r_value**2

# Regressione e grafico per i chilometri
r2_km = plot_regression(km, price, 'Chilometri', 'Prezzo', 'Regressione Lineare: km vs prezzo')
print(f"R² usando i Km (scipy): {r2_km}")

# Regressione e grafico per l'anno
r2_year = plot_regression(year, price, 'Anno', 'Prezzo', 'Regressione Lineare: anno vs prezzo')
print(f"R² usando l'anno (scipy): {r2_year}")

# Regressione e grafico per la potenza
r2_power = plot_regression(power, price, 'Potenza', 'Prezzo', 'Regressione Lineare: potenza vs prezzo')
print(f"R² usando la potenza (scipy): {r2_power}")

"""5) Then implement the least square algorithm: you should get exactly the same solution of linregress !

6) Plot the data and the lines representing the output of the linregress and least square algorithms"""

# Funzione per il metodo dei minimi quadrati (Least Squares)
def least_squares(x, y):
    A = np.vstack([x, np.ones(len(x))]).T  # Crea la matrice A con x e 1
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]  # Risolve con i minimi quadrati
    return m, b

# Funzione per tracciare il grafico con i risultati di linregress e least squares
def plot_comparison(x, y, xlabel, ylabel, title):
    # Regressione con linregress
    slope_lin, intercept_lin, _, _, _ = linregress(x, y)
    line_lin = slope_lin * x + intercept_lin

    # Regressione con Least Squares
    slope_ls, intercept_ls = least_squares(x, y)
    line_ls = slope_ls * x + intercept_ls

    # Grafico
    plt.scatter(x, y, color='blue', label='Dati Reali')
    plt.plot(x, line_lin, color='red', label='Linregress')
    plt.plot(x, line_ls, color='green', linestyle='--', label='Least Squares')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Tracciare i grafici per km, anno e potenza
plot_comparison(km, price, 'Chilometri', 'Prezzo', 'Confronto: km vs prezzo')
plot_comparison(year, price, 'Anno', 'Prezzo', 'Confronto: anno vs prezzo')
plot_comparison(power, price, 'Potenza', 'Prezzo', 'Confronto: potenza vs prezzo')