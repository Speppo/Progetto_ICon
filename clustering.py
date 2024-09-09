import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def prepara_dati_per_clustering():
    # Caricamento dei dati
    data = pd.read_csv('IMDB-Movie-Data.csv')

    # Preprocessing dei dati
    data['Genre'] = data['Genre'].apply(lambda x: x.split(', '))
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(data['Genre'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
    data = pd.concat([data, genre_df], axis=1)

    # Mantieni le colonne necessarie
    data = data[['Title', 'Director', 'Year', 'Runtime (Minutes)', 'Rating', 'Metascore'] + list(mlb.classes_)]

    # Rimuovere righe con valori NaN
    data = data.dropna()

    # Selezionare solo le colonne numeriche per la standardizzazione
    numerical_features = ['Runtime (Minutes)', 'Rating', 'Metascore']
    data_numerical = data[numerical_features]

    # Standardizzazione delle caratteristiche numeriche
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_numerical)

    return data, X_scaled


def esegui_clustering(data, X):
    # KMeans Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    data['clusterIndex'] = kmeans_labels

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    data['dbscanCluster'] = dbscan_labels

    # Salvare i dati con i cluster nei file CSV
    data.to_csv('clustered_data.csv', index=False)

    # PCA per ridurre a 2 dimensioni per la visualizzazione
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Visualizzazione dei cluster con KMeans
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', marker='o')
    plt.title('KMeans Clustering')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # Visualizzazione dei cluster con DBSCAN
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='plasma', marker='o')
    plt.title('DBSCAN Clustering')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    plt.tight_layout()
    plt.show()


# Assicurati che venga chiamata la funzione quando necessario
if __name__ == "__main__":
    data, X = prepara_dati_per_clustering()
    esegui_clustering(data, X)