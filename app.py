import streamlit as st
import pandas as pd
import pickle
import lzma
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess data for training
def preprocess_data(path):
    data = pd.read_excel(path)
    X = data.drop(columns=['TX_FRAUD','ID','TRANSACTION_ID','TX_DATETIME','TX_TIME_DAYS','CUSTOMER_ID','TERMINAL_ID','TX_FRAUD_SCENARIO', 'Column2','Column1'])
    y = data['TX_FRAUD']
    return X, y

# Function to train model
def train_model(X, y):
    # Apply SMOTE for handling imbalanced data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resampled, y_resampled)

    return model

# Function to save model
def save_model(model, file_path):
    with lzma.open(file_path, 'wb') as file:
        pickle.dump(model, file)

# Function to load model
def load_model(file_path):
    with lzma.open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    st.set_page_config(
        page_title="Prediksi Transaksi",
        layout="wide",
        page_icon="💱"
    )

    st.sidebar.title('Navigation')
    pages = ['Manual Input', 'File Upload', 'Info']
    selected = st.sidebar.radio('Go to', pages)

    if selected == 'Manual Input':
        st.title('Transaction Prediction - Manual Input')

        col1, col2 = st.columns(2)

        with col1:
            TX_AMOUNT = st.text_input('Transaction Amount', '')
        with col2:
            TX_TIME_SECONDS = st.text_input('Transaction Time Interval (Seconds)', '')

        transaction_prediction = ''

        if st.button('Transaction Prediction Result'):
            try:
                user_input = [float(TX_AMOUNT), float(TX_TIME_SECONDS)]
                prediction = loaded_model.predict([user_input])
                if prediction[0] == 1:
                    transaction_prediction = 'Your transaction is unsafe due to fraud indication'
                else:
                    transaction_prediction = 'Your transaction is safe and legitimate'
            except ValueError:
                transaction_prediction = 'Please enter valid numeric values for all inputs'

            st.success(transaction_prediction)

    elif selected == 'File Upload':
        st.title('Transaction Prediction - File Upload')

        uploaded_file = st.file_uploader("Upload Excel file with transaction data", type=["xlsx"])

        if uploaded_file is not None:
            try:
                data = pd.read_excel(uploaded_file)
                st.write("Uploaded data:")
                st.write(data)

                if 'TX_AMOUNT' in data.columns and 'TX_TIME_SECONDS' in data.columns:
                    user_inputs = data[['TX_AMOUNT', 'TX_TIME_SECONDS']].astype(float)
                    predictions = loaded_model.predict(user_inputs)

                    data['Prediction'] = predictions
                    data['Prediction'] = data['Prediction'].apply(lambda x: 'Unsafe transaction (fraud indication)' if x == 1 else 'Safe transaction')

                    st.write("Prediction Results:")
                    st.write(data)

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        data.to_excel(writer, index=False, sheet_name='Sheet1')
                    output.seek(0)

                    st.download_button(
                        label="Download prediction results",
                        data=output,
                        file_name='prediction_results.xlsx'
                    )

                    st.subheader('Transaction Time Interval Characteristics')
                    st.write(data['TX_TIME_SECONDS'].describe().to_frame().T[['mean', '50%', 'std']].rename(columns={'mean': 'Mean', '50%': 'Median', 'std': 'Std'}))

                    st.subheader('Transaction Amount Characteristics')
                    st.write(data['TX_AMOUNT'].describe().to_frame().T[['mean', '50%', 'std']].rename(columns={'mean': 'Mean', '50%': 'Median', 'std': 'Std'}))

                    st.subheader('Boxplot Transaction Time Interval')
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x=data['TX_TIME_SECONDS'], ax=ax1)
                    st.pyplot(fig1)

                    st.subheader('Boxplot Transaction Amount')
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x=data['TX_AMOUNT'], ax=ax2)
                    st.pyplot(fig2)

                else:
                    st.error('File does not have required columns: TX_AMOUNT, TX_TIME_SECONDS')
            except Exception as e:
                st.error(f"Error: {e}")

    elif selected == 'Info':
        st.title('Dashboard Information')
        
        st.write("""
        *Random Forest* is one of the commonly used machine learning algorithms for classification or prediction problems. In this case, it is used to predict which transactions fall into the fraud and legitimate classes. Predictions are based on transaction amount and transaction time interval (seconds).
        """)

        st.markdown("""
        <style>
        .center {
            display: block;
            margin-left: auto;
            margin-right: auto;
            max-width: 100%;
            height: auto;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<img src="https://cdn.prod.website-files.com/61af164800e38cf1b6c60b55/64c0c20d61bda9e68f630468_Random%20forest.webp" alt="Random Forest" width="400" class="center">', unsafe_allow_html=True)
        
        st.write("""
        There are several metrics commonly used to determine how well a model performs, including:
        
        - *Specificity* measures the model's ability to correctly identify true negatives among all actual negative cases.
        - *Sensitivity* measures the model's ability to correctly identify true positives among all actual positive cases.
        - *Accuracy* measures how often the model makes correct predictions, both for positive and negative cases.
        - *AUC ROC (Area Under the Receiver Operating Characteristic Curve)* measures the performance of a classification model at various decision thresholds.
        - *ROC (Receiver Operating Characteristic Curve)* is a graph that illustrates the true positive rate (Sensitivity) against the false positive rate (1 - Specificity) for various threshold values.
        """)

    # Load and train model
    data_path = '/content/Data_Raw.xlsx'  # Adjust path to your file location
    X, y = preprocess_data(data_path)
    loaded_model = train_model(X, y)

    # Save and load model
    save_model(loaded_model, "/content/trans_model.pkl.xz")
    loaded_model = load_model("/content/trans_model.pkl.xz")

if __name__ == "__main__":
    main()
