import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def overSampling(dataSet, clusterColumn):
    # Separazione delle caratteristiche e target
    X = dataSet.drop(columns=[clusterColumn], errors='ignore')
    y = dataSet[clusterColumn]

    # Visualizzazione della distribuzione delle classi prima dell'oversampling
    print("Distribuzione delle classi prima dell'oversampling:")
    print(y.value_counts())

    # Creazione dell'oggetto SMOTE con un numero adeguato di vicini
    min_class_size = y.value_counts().min()  # Trova la dimensione della classe minoritaria
    smote = SMOTE(random_state=42, k_neighbors=min(min_class_size - 1, 5))  # Assicurati che k_neighbors non sia maggiore di min_class_size - 1

    # Applicazione di SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Creazione del DataFrame resampled
    dataSet_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[clusterColumn])], axis=1)

    # Visualizzazione della distribuzione delle classi dopo l'oversampling
    print("Distribuzione delle classi dopo l'oversampling:")
    print(y_resampled.value_counts())

    # Funzione per tracciare i grafici a torta
    def plot_pie_chart(data, title):
        plt.figure(figsize=(10, 7))
        plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
        plt.title(title)
        plt.show()

    # Visualizzare la distribuzione delle classi prima dell'oversampling
    plot_pie_chart(y.value_counts(), 'Distribuzione delle Classi Prima dell\'Oversampling')

    # Visualizzare la distribuzione delle classi dopo l'oversampling
    plot_pie_chart(y_resampled.value_counts(), 'Distribuzione delle Classi Dopo l\'Oversampling')

    return dataSet_resampled
