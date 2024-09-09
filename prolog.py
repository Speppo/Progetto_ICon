import pandas as pd
from pyswip import Prolog

# Funzione per scrivere un fatto nel file kb.pl
def write_fact_to_file(fact, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        if not fact.endswith('.'):
            fact = fact + '.'
        file.write(f"{fact}\n")

# Funzione per scrivere i fatti temporanei
def write_temp_facts(dataSet, file_path, title_to_query=None):
    with open(file_path, "w", encoding="utf-8") as file:
        write_fact_to_file(":- encoding(utf8)", file_path)
        for _, row in dataSet.iterrows():
            # Se 'title_to_query' è fornito, carica solo il film corrispondente
            if title_to_query is None or row['Title'] == title_to_query:
                title = row['Title'].replace("'", "")
                director = row['Director'].replace("'", "")
                prolog_clause = (f"movie('{title}', '{director}', {row['Year']}, {row['Runtime (Minutes)']}, "
                                 f"{row['Rating']}, {row['Metascore']})")
                write_fact_to_file(prolog_clause, file_path)
                if 'clusterIndex' in row:
                    cluster_clause = (f"clustered_movie({row['Year']}, {row['Runtime (Minutes)']}, {row['Rating']}, "
                                      f"{row['Metascore']}, {row['clusterIndex']})")
                    write_fact_to_file(cluster_clause, file_path)

# Funzione per scrivere le regole
def write_rules(file_path):
    with open(file_path, "a", encoding="utf-8") as file:
        rule = ("movie_info(Title, Year, Cluster) :- "
                "movie(Title, _, Year, _, _, _), "
                "clustered_movie(Year, _, _, _, Cluster).")
        write_fact_to_file(rule, file_path)

# Funzione per interrogare Prolog su un film specifico
def query_prolog(title):
    prolog = Prolog()
    prolog.consult("kb_temp.pl")

    query = f"movie_info('{title}', Year, Cluster)"
    result = list(prolog.query(query))
    if result:
        for entry in result:
            print(f"Title: {title}, Year: {entry['Year']}, Cluster: {entry['Cluster']}")
    else:
        print(f"Nessun risultato trovato per il film: {title}")

if __name__ == "__main__":
    # Carica il dataset
    data = pd.read_csv('clustered_data.csv')  # Usa il dataset corretto con i cluster

    # Richiedi il nome del film all'utente
    movie_title = input("Inserisci il titolo del film da cercare: ").strip()

    # Scrivi fatti temporanei e regole in un file separato
    write_temp_facts(data, "kb_temp.pl", title_to_query=movie_title)  # Filtra per il titolo del film
    write_rules("kb_temp.pl")

    # Verifica il contenuto del file
    with open("kb_temp.pl", 'r', encoding='utf-8') as file:
        content = file.read()
        print(content)  # Stampa per verificare il contenuto

    # Interroga Prolog per il film inserito dall'utente
    query_prolog(movie_title)
