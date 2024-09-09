import subprocess
import sys


# Funzione per installare un pacchetto specifico
def install(package):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Installato Correttamente {package}")
    except subprocess.CalledProcessError as e:
        print(f"Installazione fallita {package}. Error: {e}")


# Verifica e installa tqdm se non è già installato
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm non è stato trovato. Installa tqdm...")
    install("tqdm")
    from tqdm import tqdm


# Funzione per installare le librerie necessarie
def install_packages():
    try:
        with open("libr.txt", "r") as file:
            packages = [line.strip() for line in file if line.strip()]  # Rimuovi spazi e righe vuote
    except FileNotFoundError:
        print("libr.txt non è stato trovato. Per favore assicurati che il file si trovi nella directory.")
        sys.exit(1)

    print("Installazione in corso...")
    for package in tqdm(packages):
        install(package)

    print("Tutti i packages sono stati installati correttamente")


# Esegui l'installazione dei pacchetti
if __name__ == "__main__":
    install_packages()
