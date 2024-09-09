import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from oversampling import overSampling
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def visualizza_metriche(metrics):
    models = list(metrics.keys())
    accuracy = [metrics[model]['accuracy'] for model in models]
    precision = [metrics[model]['precision'] for model in models]
    recall = [metrics[model]['recall'] for model in models]
    f1_score = [metrics[model]['f1_score'] for model in models]

    df_metrics = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    })

    print("\nMetriche dei modelli:")
    print(df_metrics)

    # Creazione dei grafici
    plt.figure(figsize=(12, 8))

    # Grafico a barre per Accuratezza
    plt.subplot(2, 2, 1)
    sns.barplot(x='Accuracy', y='Model', data=df_metrics, hue=None)
    plt.title('Accuracy per Modello')

    # Grafico a barre per Precisione
    plt.subplot(2, 2, 2)
    sns.barplot(x='Precision', y='Model', data=df_metrics, hue=None)
    plt.title('Precisione per Modello')

    # Grafico a barre per Recall
    plt.subplot(2, 2, 3)
    sns.barplot(x='Recall', y='Model', data=df_metrics, hue=None)
    plt.title('Recall per Modello')

    # Grafico a barre per F1 Score
    plt.subplot(2, 2, 4)
    sns.barplot(x='F1 Score', y='Model', data=df_metrics, hue=None)
    plt.title('F1 Score per Modello')

    plt.tight_layout()
    plt.show()


def apprendimento_supervisionato():
    print("Inizio del processo di apprendimento supervisionato...")

    # 1. Caricamento del dataset
    data = pd.read_csv('clustered_data.csv')

    # Verifica le colonne disponibili nel dataset
    print("Colonne disponibili nel dataset:", data.columns)

    # 2. Separazione delle caratteristiche e target
    target_column = 'clusterIndex'  # Aggiorna se la colonna target è diversa
    irrelevant_columns = ['Title', 'Director', 'Year', 'Rating', 'Metascore']  # Aggiorna se necessario
    if target_column not in data.columns:
        raise ValueError(f"La colonna target '{target_column}' non è presente nel dataset.")

    # Escludi le colonne non numeriche (come 'Title' e 'Director') prima di standardizzare e fare oversampling
    X = data.drop(columns=[target_column] + irrelevant_columns, errors='ignore')
    y = data[target_column]

    # 3. Rimuovere cluster con pochi campioni (es. meno di 5)
    min_samples_per_cluster = 5
    cluster_counts = y.value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_samples_per_cluster].index
    data_filtered = data[data[target_column].isin(valid_clusters)]

    # 4. Gestire valori mancanti e infiniti
    X_filtered = data_filtered.drop(columns=[target_column] + irrelevant_columns, errors='ignore')
    y_filtered = data_filtered[target_column]

    X_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)  # Sostituisci inf e -inf con NaN

    # Assicurati che solo le colonne numeriche vengano utilizzate per l'oversampling
    data_filtered_numeric = X_filtered.copy()
    data_filtered_numeric[target_column] = y_filtered

    # 5. Applicare l'oversampling
    data_filtered_resampled = overSampling(data_filtered_numeric, target_column)

    # 6. Separazione di nuove caratteristiche e target dopo l'oversampling
    X_filtered_resampled = data_filtered_resampled.drop(columns=[target_column])
    y_filtered_resampled = data_filtered_resampled[target_column]

    # 7. Divisione del dataset in set di addestramento e di test (con stratificazione)
    X_train, X_test, y_train, y_test = train_test_split(X_filtered_resampled, y_filtered_resampled, test_size=0.2,
                                                        random_state=42,
                                                        stratify=y_filtered_resampled)

    # 8. Standardizzazione delle caratteristiche
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 9. Definizione dei modelli di classificazione e degli iperparametri
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(probability=True),
        'KNeighbors': KNeighborsClassifier()
    }

    param_grid = {
        'RandomForest': {
            'n_estimators': [50, 100],  # Ridotto numero di estimatori
            'max_depth': [5, 10],  # Limitata la profondità
            'min_samples_split': [2, 5]
        },
        'SVC': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        'KNeighbors': {
            'n_neighbors': [3, 5],
            'weights': ['uniform']
        }
    }

    # 10. Ricerca degli iperparametri e addestramento dei modelli
    best_models = {}
    for model_name in models:
        print(f"Addestramento del modello: {model_name}")
        grid_search = GridSearchCV(
            models[model_name],
            param_grid[model_name],
            cv=KFold(n_splits=3, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        print(f"Migliori parametri per {model_name}: {grid_search.best_params_}")
        print(f"Accuracy migliore su CV per {model_name}: {grid_search.best_score_}\n")

    # 11. Valutazione dei modelli sui dati di test
    metrics = {}
    for model_name in best_models:
        y_pred = best_models[model_name].predict(X_test)
        y_pred_proba = best_models[model_name].predict_proba(X_test)[:, 1]  # Probabilità della classe positiva

        print(f"Valutazione del modello: {model_name}")
        print(f"Accuracy su test: {accuracy_score(y_test, y_pred)}")

        # Verifica la distribuzione delle etichette reali e predette
        print("Distribuzione delle etichette reali:", np.bincount(y_test))
        print("Distribuzione delle predizioni:", np.bincount(y_pred))

        # Confusion Matrix e Classification Report
        print(f"Confusion Matrix per {model_name}:\n {confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report per {model_name}:\n {classification_report(y_test, y_pred, zero_division=0)}\n")

        # Confronta l'accuratezza su training e test per controllare l'overfitting
        train_accuracy = accuracy_score(y_train, best_models[model_name].predict(X_train))
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy su training per {model_name}: {train_accuracy}")
        print(f"Accuracy su test per {model_name}: {test_accuracy}\n")

        # Salvo le metriche per la visualizzazione
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted',
                                                                       zero_division=0)
        metrics[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision,
            'recall': recall,
            'f1_score': fscore
        }

        # Grafico delle curve di apprendimento
        train_sizes, train_scores, test_scores = learning_curve(
            best_models[model_name], X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
        plt.title(f'Learning Curves ({model_name})')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

        # Controllo della varianza e deviazione standard
        print(f"\nTraining Data Variance per {model_name}: {np.var(y_train):.2f}")
        print(f"Training Data Standard Deviation per {model_name}: {np.std(y_train):.2f}")
        print(f"Test Data Variance per {model_name}: {np.var(y_test):.2f}")
        print(f"Test Data Standard Deviation per {model_name}: {np.std(y_test):.2f}")

    # 12. Visualizzazione dei risultati
    visualizza_metriche(metrics)

    # 13. Salvataggio del miglior modello
    best_model_name = max(best_models, key=lambda k: accuracy_score(y_test, best_models[k].predict(X_test)))
    joblib.dump(best_models[best_model_name], 'best_classification_model.pkl')
    print(f"Il miglior modello ({best_model_name}) è stato salvato come 'best_classification_model.pkl'")

if __name__ == "__main__":
    apprendimento_supervisionato()
