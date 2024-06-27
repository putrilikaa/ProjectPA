import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import lzma
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter  # Mengganti openpyxl dengan xlsxwriter

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
    selected = option_menu(
        'Prediksi Transaksi',
        [
            'Manual Input',
            'File Upload',
            'Info'
        ],
        menu_icon='money-fill',
        icons=['pencil', 'upload', 'info-circle'],
        default_index=0
    )

# Halaman input manual
if selected == 'Manual Input':
    st.title('Transaction Prediction - Manual Input')

    col1, col2 = st.columns(2)

    with col1:
        TX_AMOUNT = st.text_input('Jumlah Transaksi', '')  # Mengganti label dengan Jumlah Transaksi
    with col2:
        TX_TIME_SECONDS = st.text_input('Jeda Waktu Transaksi (Detik)', '')  # Mengganti label dengan Jeda Waktu Transaksi (Detik)

    transaction_prediction = ''

    if st.button('Transaction Prediction Result'):
        try:
            user_input = [float(TX_AMOUNT), float(TX_TIME_SECONDS)]
            transaction_diagnosis = trans_model.predict([user_input])
            if transaction_diagnosis[0] == 1:
                transaction_prediction = 'Transaksi anda tidak aman karena terjadi indikasi penipuan'
            else:
                transaction_prediction = 'Transaksi anda aman karena dilakukan secara sah'
        except ValueError:
            transaction_prediction = 'Harap masukkan nilai numerik yang valid untuk semua input'
        
        st.success(transaction_prediction)

# Halaman upload file
elif selected == 'File Upload':
    st.title('Transaction Prediction - File Upload')

    uploaded_file = st.file_uploader("Upload file Excel dengan data transaksi", type=["xlsx"])

    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.write("Data yang diupload:")
            st.write(data)

            if 'TX_AMOUNT' in data.columns and 'TX_TIME_SECONDS' in data.columns:
                user_inputs = data[['TX_AMOUNT', 'TX_TIME_SECONDS']].astype(float)
                predictions = trans_model.predict(user_inputs)

                data['Prediction'] = predictions
                data['Prediction'] = data['Prediction'].apply(lambda x: 'Transaksi tidak aman (indikasi penipuan)' if x == 1 else 'Transaksi aman')

                st.write("Hasil Prediksi:")
                st.write(data)

                # Menampilkan tabel statistik deskriptif
                st.subheader('Karakteristik Jeda Waktu Detik')
                st.write(data['TX_TIME_SECONDS'].describe().to_frame().T[['mean', '50%', 'std']].rename(columns={'mean': 'Rata-Rata', '50%': 'Median', 'std': 'Varians'}))

                st.subheader('Karakteristik Jumlah Transaksi')
                st.write(data['TX_AMOUNT'].describe().to_frame().T[['mean', '50%', 'std']].rename(columns={'mean': 'Rata-Rata', '50%': 'Median', 'std': 'Varians'}))

                # Mengkonversi DataFrame ke Excel menggunakan xlsxwriter tanpa engine_kwargs
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    data.to_excel(writer, index=False, sheet_name='Sheet1')
                
                output.seek(0)

                st.download_button(
                    label="Download hasil prediksi",
                    data=output,
                    file_name='hasil_prediksi.xlsx'
                )

            else:
                st.error('File tidak memiliki kolom yang diperlukan: TX_AMOUNT, TX_TIME_SECONDS')
        except Exception as e:
            st.error(f"Error: {e}")

# Halaman informasi
elif selected == 'Info':
    st.title('Informasi Dashboard')
    
    st.write("""
    *Random Forest* adalah salah satu algoritma machine learning yang umum digunakan dalam permasalahan klasifikasi atau prediksi. Pada kasus ini digunakan untuk memprediksi mana transaksi yang termasuk ke dalam kelas penipuan dan sah. Prediksi didasarkan pada jumlah transaksi dan jeda waktu transaksi (detik).
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
    
    - *Spesifisitas (Specificity)* mengukur kemampuan model untuk dengan benar mengidentifikasi negatif sejati (true negatives) di antara semua kasus yang sebenarnya negatif.
    - *Sensitivitas (Sensitivity)* mengukur kemampuan model untuk dengan benar mengidentifikasi positif sejati (true positives) di antara semua kasus yang sebenarnya positif.
    - *Akurasi (Accuracy)* mengukur seberapa sering model membuat prediksi yang benar, baik untuk kasus positif maupun negatif.
    - *AUC ROC (Area Under the Receiver Operating Characteristic Curve)* mengukur kinerja model klasifikasi pada berbagai threshold keputusan.
    - *ROC (Receiver Operating Characteristic Curve)* adalah grafik yang menggambarkan rasio True Positive Rate (Sensitivitas) terhadap False Positive Rate (1 - Spesifisitas) untuk berbagai nilai threshold.
    """)

