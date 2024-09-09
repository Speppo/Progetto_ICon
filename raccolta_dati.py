import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


def load_data():
    data = pd.read_csv('IMDB-Movie-Data.csv')
    print(data['Genre'].head())  # Stampa i primi valori della colonna 'Genre' per verificare il formato
    return data


# Funzione di preprocessing dei dati
def preprocess_data(data):
    # Convertire i generi in una lista
    data['Genre'] = data['Genre'].apply(lambda x: x.split(', '))

    # Binarizzazione dei generi
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(data['Genre'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

    # Concatenare il dataframe originale con il dataframe dei generi
    data = pd.concat([data, genre_df], axis=1)

    # Selezionare le colonne di interesse
    data = data[['Title', 'Director', 'Year', 'Runtime (Minutes)', 'Rating', 'Metascore'] + list(mlb.classes_)]

    # Rimuovere righe con valori mancanti
    data = data.dropna()

    return data


# Caricamento dei dati dal CSV
data = pd.read_csv('IMDB-Movie-Data.csv')

# Preprocessing dei dati
processed_data = preprocess_data(data)

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


# Esempio di test per verificare la codifica dei generi
def test_genre_encoding():
    data = pd.DataFrame({
        'Genre': ["['Action', 'Adventure']", "['Adventure', 'Fantasy']"]
    })

    data['Genre'] = data['Genre'].apply(lambda x: x.strip("[]").replace("'", "").split(", "))

    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(data['Genre'])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

    print("Esempio di codifica dei generi:")
    print(genres_df)

def raccogli_dati():
    # Logica di raccolta dati
    print("Raccolta dati completata")


if __name__ == "__main__":
    # Esegui il test della codifica dei generi
    test_genre_encoding()

    # Carica e preprocessa i dati
    data = load_data()
    processed_data = preprocess_data(data)
    processed_data.to_csv('processed_data.csv', index=False)

    # Standardizzazione delle features numeriche
    numerical_features = ['Runtime (Minutes)', 'Rating', 'Metascore']
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(processed_data[numerical_features])

    # Creazione di un DataFrame standardizzato
    data_standardized_df = pd.DataFrame(data_standardized, columns=numerical_features)

    # Concatenazione con le colonne binarizzate dei generi
    final_data = pd.concat([processed_data[['Title', 'Year']], data_standardized_df,
                            processed_data.drop(columns=['Title', 'Year', 'Runtime (Minutes)', 'Rating', 'Metascore'])],
                           axis=1)

    # Salvataggio
    final_data.to_csv('data_standardized.csv', index=False)

    print("Dati preprocessati e standardizzati salvati correttamente.")

    # Raccolta dei dati
    raccogli_dati()
