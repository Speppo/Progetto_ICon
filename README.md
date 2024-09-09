# Progetto_ICon
Questo progetto applica tecniche di machine learning e clustering per analizzare e classificare un dataset di film tratto da IMDb. Utilizza metodi di apprendimento non supervisionato per raggruppare i film in cluster e apprendimento supervisionato per prevedere a quale cluster appartiene un film. Inoltre, è integrato con Prolog per consentire interrogazioni logiche sui dati dei film e dei cluster.

# Descrizione del Progetto
Il progetto si concentra su:

Clustering dei film: Utilizzo di algoritmi di clustering per raggruppare i film in base alle loro caratteristiche (durata, rating, metascore, ecc.).  
Apprendimento supervisionato: Addestramento di modelli di classificazione (come Random Forest e SVM) per prevedere il cluster a cui appartiene un film.  
Integrazione con Prolog: Implementazione di un sistema di interrogazione logica basato su Prolog, che permette di effettuare query sui film e sui loro cluster.  

# Tecnologie utilizzate
-Python 3.12  
-Pandas e NumPy per la manipolazione dei dati.  
-Scikit-learn per il clustering e l'apprendimento supervisionato.  
-Matplotlib e Seaborn per la visualizzazione dei dati.  
-Imbalanced-learn (SMOTE) per bilanciare le classi.  
-PySWIP per l'integrazione con Prolog.  
-Joblib per salvare e caricare i modelli addestrati.  

# Come installare e utilizzare il progetto  
Prerequisiti  
Assicurati di avere Python 3.12 installato. Inoltre, ti serviranno le seguenti librerie Python, che puoi installare tramite pip:  
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn pyswip joblib


# Struttura del progetto
-install.py: installa tutte le librerie necessarie per l'esecuzione del progetto.  
-main.py: Coordina tutte le fasi del progetto: preprocessing, clustering, apprendimento supervisionato e interrogazioni Prolog.  
-raccolta_dati.py: Esegue la raccolta e il preprocessing dei dati (binarizzazione dei generi, standardizzazione delle feature numeriche).  
-clustering.py: Esegue il clustering dei film utilizzando KMeans e DBSCAN.  
-supervised_learning.py: Addestra i modelli di classificazione per prevedere il cluster di appartenenza di un film.  
-prolog.py: Genera fatti e regole per Prolog e permette di eseguire query logiche sui film.  

# Caratteristiche principali
-Clustering con KMeans e DBSCAN: Suddivide i film in cluster in base alle caratteristiche.  
-Apprendimento supervisionato: Utilizza modelli di classificazione come Random Forest, SVC e KNN.  
-Bilanciamento delle classi con SMOTE: Evita lo sbilanciamento tra i cluster usando tecniche di oversampling.  
-Integrazione con Prolog: Consente interrogazioni logiche sui film e sui cluster attraverso una base di conoscenza generata dinamicamente.  
