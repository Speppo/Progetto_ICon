import subprocess

def main():
    print("Inizio della raccolta dei dati e preprocessing...")
    subprocess.run(["python", "raccolta_dati.py"], check=True)
    print("Raccolta dei dati e preprocessing completati.")

    print("Inizio del clustering...")
    subprocess.run(["python", "clustering.py"], check=True)
    print("Clustering completato.")

    print("Inizio dell'apprendimento supervisionato...")
    subprocess.run(["python", "supervised_learning.py"], check=True)
    print("Apprendimento supervisionato completato.")

    print("Inizio della creazione e interrogazione della base di conoscenza Prolog...")
    subprocess.run(["python", "prolog.py"], check=True)
    print("Creazione e interrogazione della base di conoscenza Prolog completate.")

if __name__ == "__main__":
    main()
