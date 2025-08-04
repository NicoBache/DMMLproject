import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt

def loan_feature_descriptor(df, exclude_cols=['TARGET'], verbose=True):
    '''
    Fornisce un riassunto delle variabili numeriche:
    - Statistiche descrittive (mean, std, min, max)
    - Percentuale di valori mancanti
    - Skewness (asimmetria) della distribuzione

    Parametri:
    - df: DataFrame in input
    - exclude_cols: lista di colonne da escludere (default = ['TARGET'])
    - verbose: se True stampa tutte le metriche, altrimenti solo le skew

    Utile per EDA (analisi esplorativa) prima del preprocessing.
    '''
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    if verbose:
        print("==== STATISTICHE DESCRITTIVE ====")
        print(df[numeric_cols].describe().T)

        print("\n==== PERCENTUALI DI MISSING ====")
        missing = df[numeric_cols].isnull().mean().sort_values(ascending=False)
        print((missing * 100).round(2)[missing > 0])

    print("\n==== SKEWNESS (Asimmetria) ====")
    skew_vals = df[numeric_cols].apply(lambda x: skew(x.dropna()))
    print(skew_vals.sort_values(ascending=False).round(2).head(10))


def check_skew_and_log_effect(df, col):
    '''
    Mostra visivamente la distribuzione di una feature continua
    prima e dopo trasformazione logaritmica.

    Utile per identificare asimmetrie forti da correggere.
    
    Scopo della funzione
    Capire se la variabile è fortemente sbilanciata (asimmetrica o skewed)

    Valutare visivamente se applicare una trasformazione logaritmica migliora la distribuzione

    Aiutarti a decidere se trasformare quella feature prima di usarla in un modello di classificazione



    Parametri:
    - df: DataFrame
    - col: nome della colonna da analizzare
    '''
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    df[col].dropna().hist(ax=ax[0], bins=30, color='skyblue')
    ax[0].set_title(f"Distribuzione originale: {col}")

    np.log1p(df[col].dropna()).hist(ax=ax[1], bins=30, color='salmon')
    ax[1].set_title(f"Distribuzione dopo log1p: {col}")

    plt.tight_layout()
    plt.show()




# Suggests which columns to apply log1p transformation based on skewness
# Returns a list of columns that are skewed enough to benefit from log transformation
# Prints the skewness before and after transformation for each column
def suggest_log_transform(df, threshold=1.0, verbose=True):
    """
    Analizza le variabili numeriche del DataFrame e suggerisce su quali applicare log1p
    per ridurre la skewness (asimmetria).

    Parametri:
    - df: DataFrame
    - threshold: soglia di skewness oltre la quale si considera la trasformazione
    - verbose: se True stampa le variabili da trasformare

    Output:
    - Ritorna una lista con i nomi delle colonne consigliate per log-transform
    """
    cols_to_log = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.min() < 0:
            continue  # non loggo variabili con valori negativi

        original_skew = skew(series)
        log_skew = skew(np.log1p(series))

        if original_skew >= threshold and log_skew < original_skew:
            cols_to_log.append(col)
            if verbose:
                print(f" {col}: skew originale = {original_skew:.2f}, dopo log1p = {log_skew:.2f} → ✔ consigliata log-transform")

    if verbose and not cols_to_log:
        print(" Nessuna feature ha skewness sufficiente per richiedere log-transform.")

    return cols_to_log


