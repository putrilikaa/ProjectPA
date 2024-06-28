import streamlit as st
import pickle
import lzma
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Prediksi Transaksi",
    layout="wide",
    page_icon="💱"
)

# Fungsi untuk memuat model terkompresi
@st.cache(allow_output_mutation=True)
def load_compressed_model(file_path):
    try:
        with lzma.open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"File '{file_path}' tidak ditemukan. Mohon pastikan file model tersedia di lokasi yang benar.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# Memuat model yang disimpan
model_file = 'trans_model.pkl.xz'  # Ubah sesuai dengan nama file model Anda
trans_model = load_compressed_model(model_file)

if trans_model is None:
    st.stop()

# Sidebar untuk navigasi
with st.sidebar:
    selected = st.selectbox(
        'Navigation',
        ['Manual Input', 'File Upload', 'Random Forest Modeling', 'Info']
    )

# Halaman input manual
if selected == 'Manual Input':
    st.title('Transaction Prediction - Manual Input')

    col1, col2 = st.columns(2)

    with col1:
        TX_AMOUNT = st.text_input('TX_AMOUNT', '')
    with col2:
        TX_TIME_SECONDS = st.text_input('TX_TIME_SECONDS', '')

    transaction_prediction = ''

    if st.button('Transaction Prediction Result'):
        try:
            user_input = [[float(TX_AMOUNT), float(TX_TIME_SECONDS)]]
            transaction_diagnosis = trans_model.predict(user_input)
            if transaction_diagnosis[0] == 1:
                transaction_prediction = 'Fraud'
            else:
                transaction_prediction = 'Not Fraud'
        except ValueError:
            transaction_prediction = 'Please enter valid numeric values for all inputs'

        st.success(f'Transaction is predicted as: {transaction_prediction}')

# Halaman upload file
elif selected == 'File Upload':
    st.title('Transaction Prediction - File Upload')

    st.markdown("Upload an Excel file containing 'TX_AMOUNT' and 'TX_TIME_SECONDS' columns.")

    uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])

    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)

            if 'TX_AMOUNT' in data.columns and 'TX_TIME_SECONDS' in data.columns:
                predictions = trans_model.predict(data[['TX_AMOUNT', 'TX_TIME_SECONDS']])

                data['Prediction'] = ['Fraud' if pred == 1 else 'Not Fraud' for pred in predictions]

                st.write(data)

                # Download link for predictions
                output = io.BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                data.to_excel(writer, index=False, sheet_name='Predictions')
                writer.save()
                output.seek(0)
                st.download_button(label="Download Predictions", data=output, file_name='predictions.xlsx')

            else:
                st.error("File does not have required columns: 'TX_AMOUNT' and 'TX_TIME_SECONDS'.")

        except Exception as e:
            st.error(f"Error: {e}")

# Halaman pemodelan Random Forest
elif selected == 'Random Forest Modeling':
    st.title('Random Forest Modeling')

    st.write("""
    This page is used to evaluate the model using Random Forest modeling. The data used here is unrelated to the data uploaded on the previous page.
    """)

    st.markdown("Upload an Excel file containing 'TX_AMOUNT', 'TX_TIME_SECONDS', and 'TX_FRAUD' columns.")

    uploaded_file_rf = st.file_uploader("Choose a file", type=["xlsx"])

    if uploaded_file_rf is not None:
        try:
            data_rf = pd.read_excel(uploaded_file_rf)

            if 'TX_AMOUNT' in data_rf.columns and 'TX_TIME_SECONDS' in data_rf.columns and 'TX_FRAUD' in data_rf.columns:
                user_inputs_rf = data_rf[['TX_AMOUNT', 'TX_TIME_SECONDS']].astype(float)
                true_labels_rf = data_rf['TX_FRAUD'].astype(int)
                predictions_rf = trans_model.predict(user_inputs_rf)

                data_rf['Prediction'] = ['Fraud' if pred == 1 else 'Not Fraud' for pred in predictions_rf]

                st.write(data_rf)

                # Evaluation metrics
                st.subheader('Model Evaluation Metrics')

                accuracy_rf = accuracy_score(true_labels_rf, predictions_rf)
                st.write(f'Accuracy: {accuracy_rf:.2f}')

                auc_rf = roc_auc_score(true_labels_rf, predictions_rf)
                st.write(f'AUC ROC: {auc_rf:.2f}')

                # Confusion matrix
                st.subheader('Confusion Matrix')
                cm_rf = confusion_matrix(true_labels_rf, predictions_rf)
                st.write(cm_rf)

                # Calculate Sensitivity and Specificity
                TN = cm_rf[0, 0]
                FP = cm_rf[0, 1]
                FN = cm_rf[1, 0]
                TP = cm_rf[1, 1]

                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)

                st.write(f'Sensitivity: {sensitivity:.2f}')
                st.write(f'Specificity: {specificity:.2f}')

                # Download link for predictions and evaluation
                output_rf = io.BytesIO()
                writer_rf = pd.ExcelWriter(output_rf, engine='xlsxwriter')
                data_rf.to_excel(writer_rf, index=False, sheet_name='Predictions')
                writer_rf.save()
                output_rf.seek(0)
                st.download_button(label="Download Predictions and Evaluation", data=output_rf, file_name='predictions_evaluation.xlsx')

            else:
                st.error("File does not have required columns: 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_FRAUD'.")

        except Exception as e:
            st.error(f"Error: {e}")

# Halaman informasi
elif selected == 'Info':
    st.title('Dashboard Information')

    st.write("""
    This dashboard uses modeling with the ***Random Forest*** algorithm, which is commonly used in classification or prediction problems. In this case, it is used to predict which transactions belong to the fraud and non-fraud classes. Predictions are based on transaction amount and transaction time gap (seconds).
    """)

    st.markdown("""
    ### Key Performance Metrics:
    - **Specificity (True Negative Rate)** measures the model's ability to correctly identify true negatives among all cases that are actually negative.
    - **Sensitivity (True Positive Rate)** measures the model's ability to correctly identify true positives among all cases that are actually positive.
    - **Accuracy** measures how often the model makes correct predictions, both for positive and negative cases.
    - **AUC ROC (Area Under the Receiver Operating Characteristic Curve)** measures the performance of a classification model at various thresholds.
    - **ROC (Receiver Operating Characteristic Curve)** is a graph that illustrates the ratio of True Positive Rate (Sensitivity) to False Positive Rate (1 - Specificity) for various threshold values.
    """)

    st.markdown("""
    ![Random Forest](https://cdn.prod.website-files.com/61af164800e38cf1b6c60b55/64c0c20d61bda9e68f630468_Random%20forest.webp)
    """)

