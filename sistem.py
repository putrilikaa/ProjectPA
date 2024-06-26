import pandas as pd
import pickle
import lzma
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def train_model(data):
    # Preprocess data
    X = data.drop(columns=['TX_FRAUD','ID','TRANSACTION_ID','TX_DATETIME','TX_TIME_DAYS','CUSTOMER_ID','TERMINAL_ID','TX_FRAUD_SCENARIO', 'Column2','Column1'])
    y = data['TX_FRAUD']

    # Apply SMOTE for handling imbalanced data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resampled, y_resampled)

    return model

def save_model(model, file_path):
    with lzma.open(file_path, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_path):
    with lzma.open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # Load data for training
    path = '/content/Data_Raw.xlsx'  # Sesuaikan path dengan lokasi file Anda
    data = pd.read_excel(path)

    # Train model
    model = train_model(data)

    # Save model
    save_model(model, "/content/trans_model.pkl.xz")

    # Load model
    loaded_model = load_model("/content/trans_model.pkl.xz")

if __name__ == "__main__":
    main()
