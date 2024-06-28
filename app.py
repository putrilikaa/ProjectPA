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

# Custom CSS to style the sidebar
st.markdown("""
    <style>
    .css-1d391kg {
        background-color: #F8F9FA !important;
    }
    .css-1d391kg .css-1d391kg {
        color: black !important;
    }
    .css-10trblm {
        background-color: #ffffff !important;
    }
    .css-1d391kg .st-af {
        color: #ffffff;
        background-color: #FF4B4B !important;
    }
    .css-1d391kg .st-ae {
        color: black !important;
    }
    .css-1d391kg .st-ae:hover {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat model terkompresi
@st.cache_resource
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
    selected = st.radio(
        'Main Menu',
        options=[
            'Dashboard',
            'Data Visualization',
            'Pemodelan Random Forest',
            'Info'
        ],
        format_func=lambda x: 'Dashboard' if x == 'Dashboard' else ('Data Visualization' if x == 'Data Visualization' else ('Pemodelan Random Forest' if x == 'Pemodelan Random Forest' else 'Info'))
    )

# Halaman Dashboard
if selected == 'Dashboard':
    st.title('Dashboard')

    # Your dashboard content here

# Halaman Data Visualization
elif selected == 'Data Visualization':
    st.title('Data Visualization')

    # Your data visualization content here

# Halaman pemodelan Random Forest
elif selected == 'Pemodelan Random Forest':
    st.title('Pemodelan Random Forest')

    st.write("""
    Halaman ini digunakan untuk evaluasi model menggunakan pemodelan Random Forest. Data yang digunakan di sini tidak terkait dengan data yang diupload pada halaman sebelumnya.
    """)

    st.markdown("**Upload file excel yang berisi data TX_AMOUNT, TX_TIME_SECONDS dan TX_FRAUD**")
    st.markdown("**TX_AMOUNT:** Data jumlah transaksi")
    st.markdown("**TX_TIME_SECONDS:** Data jeda waktu transaksi dalam detik")
    st.markdown("**TX_FRAUD:** Status transaksi 1 (Penipuan) dan 0 (Sah)")
    st.markdown("**NOTE:** Beri nama kolom sesuai keterangan di atas dan gunakan tanda titik (.) sebagai koma (,)")

    uploaded_file_rf = st.file_uploader("", type=["xlsx"])

    if uploaded_file_rf is not None:
        try:
            data_rf = pd.read_excel(uploaded_file_rf)
            st.write("Data yang diupload untuk evaluasi model:")
            st.write(data_rf)

            if 'TX_AMOUNT' in data_rf.columns and 'TX_TIME_SECONDS' in data_rf.columns and 'TX_FRAUD' in data_rf.columns:
                if 'Prediction' in data_rf.columns:
                    data_rf.drop(columns=['Prediction'], inplace=True)

                user_inputs_rf = data_rf[['TX_AMOUNT', 'TX_TIME_SECONDS']].astype(float)
                true_labels_rf = data_rf['TX_FRAUD'].astype(int)
                predictions_rf = trans_model.predict(user_inputs_rf)

                data_rf['Prediction'] = predictions_rf
                data_rf['Prediction'] = data_rf['Prediction'].apply(lambda x: 'Transaksi tidak aman (indikasi penipuan)' if x == 1 else 'Transaksi aman')

                st.write("Hasil Prediksi:")
                st.write(data_rf)

                # Menampilkan metrik evaluasi
                st.subheader('Metrik Evaluasi Model')

                accuracy_rf = accuracy_score(true_labels_rf, predictions_rf)
                st.write(f'**Akurasi**: {accuracy_rf:.2f}')

                auc_rf = roc_auc_score(true_labels_rf, predictions_rf)
                st.write(f'**AUC ROC**: {auc_rf:.2f}')

                # Menampilkan Confusion Matrix
                st.subheader('Confusion Matrix')
                cm_rf = confusion_matrix(true_labels_rf, predictions_rf)
                
                # Tampilkan keterangan di bawah Confusion Matrix
                st.markdown("""
                ### Penjelasan Confusion Matrix:
                - **True Positive (TP)**: Transaksi yang sebenarnya penipuan dan diprediksi sebagai penipuan.
                - **True Negative (TN)**: Transaksi yang sebenarnya sah dan diprediksi sebagai sah.
                - **False Positive (FP)**: Transaksi yang sebenarnya sah tetapi diprediksi sebagai penipuan.
                - **False Negative (FN)**: Transaksi yang sebenarnya penipuan tetapi diprediksi sebagai sah.
                """)

                plt.figure(figsize=(6, 4))
                sns.heatmap(cm_rf, annot=True, cmap='Reds', fmt='g')  # fmt='g' untuk menampilkan angka tanpa desimal jika angka integer
                plt.xlabel('Prediksi')
                plt.ylabel('Aktual')
                st.pyplot()

                st.markdown(f"Transaksi yang sebenarnya sah dan diprediksi sebagai sah adalah sebesar {cm_rf[0, 0]} data")
                st.markdown(f"Transaksi yang sebenarnya sah tetapi diprediksi sebagai penipuan adalah sebesar {cm_rf[0, 1]} data")
                st.markdown(f"Transaksi yang sebenarnya penipuan dan diprediksi sebagai penipuan adalah sebesar {cm_rf[1, 1]} data")
                st.markdown(f"Transaksi yang sebenarnya penipuan tetapi diprediksi sebagai sah adalah sebesar {cm_rf[1, 0]} data")

                # Mengkonversi DataFrame ke Excel menggunakan xlsxwriter tanpa engine_kwargs
                output_rf = io.BytesIO()
                with pd.ExcelWriter(output_rf, engine='xlsxwriter') as writer:
                    data_rf.to_excel(writer, index=False, sheet_name='Sheet1')
                
                output_rf.seek(0)

                st.download_button(
                    label="Download hasil prediksi dan evaluasi",
                    data=output_rf,
                    file_name='hasil_prediksi_dan_evaluasi.xlsx'
                )

            else:
                st.error('File tidak memiliki kolom yang diperlukan: TX_AMOUNT, TX_TIME_SECONDS, TX_FRAUD')
        except Exception as e:
            st.error(f"Error: {e}")

# Halaman informasi
elif selected == 'Info':
    st.title('Informasi Dashboard')
    
    st.write("""
    Dashboard ini menggunakan pemodelan dengan algoritma ***Random Forest*** yang merupakan salah satu algoritma machine learning yang umum digunakan dalam permasalahan klasifikasi atau prediksi. Pada kasus ini digunakan untuk memprediksi mana transaksi yang termasuk ke dalam kelas penipuan dan sah. Prediksi didasarkan pada jumlah transaksi dan jeda waktu transaksi (detik).
    """)

    # Menampilkan gambar Random Forest dengan st.image dan mengatur penempatan dengan CSS
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
    Terdapat beberapa pengukuran yang biasa digunakan untuk menentukan seberapa baik model, antara lain:
    
    - **Spesifisitas (Specificity)** mengukur kemampuan model untuk dengan benar mengidentifikasi negatif sejati (true negatives) di antara semua kasus yang sebenarnya negatif.
    - **Sensitivitas (Sensitivity)** mengukur kemampuan model untuk dengan benar mengidentifikasi positif sejati (true positives) di antara semua kasus yang sebenarnya positif.
    - **Akurasi (Accuracy)** mengukur seberapa sering model membuat prediksi yang benar, baik untuk kasus positif maupun negatif.
    - **AUC ROC (Area Under the Receiver Operating Characteristic Curve)** mengukur kinerja model klasifikasi pada berbagai threshold keputusan.
    - **ROC (Receiver Operating Characteristic Curve)** adalah grafik yang menggambarkan rasio True Positive Rate (Sensitivitas) terhadap False Positive Rate (1 - Spesifisitas) untuk berbagai nilai threshold.
    """)
