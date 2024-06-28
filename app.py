import streamlit as st
import pickle
import lzma
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Prediksi Transaksi",
    layout="wide",
    page_icon="💱"
)

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
    selected = st.selectbox(
        'Navigasi',
        ['Manual Input', 'File Upload', 'Pemodelan Random Forest', 'Info'],
        index=2
    )

# Halaman input manual
if selected == 'Manual Input':
    st.title('Prediksi Transaksi - Input Manual')

    col1, col2 = st.columns(2)

    with col1:
        TX_AMOUNT = st.text_input('Jumlah Transaksi', '')
    with col2:
        TX_TIME_SECONDS = st.text_input('Jeda Waktu Transaksi (Detik)', '')

    transaction_prediction = ''

    if st.button('Hasil Prediksi'):
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

# Halaman upload file (untuk transaksi biasa)
elif selected == 'File Upload':
    st.title('Prediksi Transaksi - Upload File Excel')

    st.write("Halaman ini digunakan untuk memprediksi status transaksi sah atau terindikasi penipuan")

    st.write("**TX_AMOUNT**: Jumlah transaksi")
    st.write("**TX_TIME_SECONDS**: Jeda waktu transaksi dalam detik")
    st.write("**Note**: File harus berformat excel")

    uploaded_file = st.file_uploader('Upload file excel yang berisi data TX_AMOUNT dan TX_TIME_SECONDS', type=["xlsx"])

    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.write("Data yang diupload untuk prediksi transaksi biasa:")
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

                # Dropdown untuk memilih tipe plot
                plot_type = st.selectbox('**Pilih jenis plot:**', ['Histogram', 'Boxplot'])

                if plot_type == 'Histogram':
                    # Menampilkan histogram
                    st.subheader('Distribusi Jumlah Transaksi')
                    fig_hist, ax_hist = plt.subplots()
                    sns.histplot(data['TX_AMOUNT'], kde=True, ax=ax_hist, color='lightcoral')
                    ax_hist.set_xlabel('TX_AMOUNT')
                    ax_hist.set_ylabel('Frekuensi')
                    st.pyplot(fig_hist)

                    st.subheader('Distribusi Jeda Waktu Transaksi (Detik)')
                    fig_hist_time, ax_hist_time = plt.subplots()
                    sns.histplot(data['TX_TIME_SECONDS'], kde=True, ax=ax_hist_time, color='lightcoral')
                    ax_hist_time.set_xlabel('TX_TIME_SECONDS')
                    ax_hist_time.set_ylabel('Frekuensi')
                    st.pyplot(fig_hist_time)

                elif plot_type == 'Boxplot':
                    # Menampilkan boxplot
                    st.subheader('Boxplot Jumlah Transaksi')
                    fig_box, ax_box = plt.subplots()
                    sns.boxplot(data['TX_AMOUNT'], ax=ax_box, color='lightcoral')
                    ax_box.set_xlabel('TX_AMOUNT')
                    st.pyplot(fig_box)

                    st.subheader('Boxplot Jeda Waktu Transaksi (Detik)')
                    fig_box_time, ax_box_time = plt.subplots()
                    sns.boxplot(data['TX_TIME_SECONDS'], ax=ax_box_time, color='lightcoral')
                    ax_box_time.set_xlabel('TX_TIME_SECONDS')
                    st.pyplot(fig_box_time)

                # Mengkonversi DataFrame ke Excel menggunakan xlsxwriter
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

# Halaman pemodelan Random Forest
elif selected == 'Pemodelan Random Forest':
    st.title('Pemodelan Random Forest')

    st.write("Halaman ini digunakan untuk evaluasi model dengan menggunakan algoritma ***Random Forest***")

    st.write("**TX_AMOUNT**: Jumlah transaksi")
    st.write("**TX_TIME_SECONDS**: Jeda waktu transaksi dalam detik")
    st.write("**TX_FRAUD**: Status transaksi (0 untuk sah, 1 untuk penipuan)")
    st.write("**Note**: File harus berformat excel")

    uploaded_file_rf = st.file_uploader('Upload file excel yang berisi data TX_AMOUNT, TX_TIME_SECONDS, dan TX_FRAUD', type=["xlsx"])

    if uploaded_file_rf is not None:
        try:
            data_rf = pd.read_excel(uploaded_file_rf)
            st.write("Data yang diupload untuk evaluasi model:")
            st.write(data_rf)

            if 'TX_AMOUNT' in data_rf.columns and 'TX_TIME_SECONDS' in data_rf.columns and 'TX_FRAUD' in data_rf.columns:
                user_inputs_rf = data_rf[['TX_AMOUNT', 'TX_TIME_SECONDS']].astype(float)
                true_labels_rf = data_rf['TX_FRAUD'].astype(int)
                predictions_rf = trans_model.predict(user_inputs_rf)

                data_rf['Prediction'] = predictions_rf
                data_rf['Prediction'] = data_rf['Prediction'].apply(lambda x: 'Transaksi tidak aman (indikasi penipuan)' if x == 1 else 'Transaksi aman')

                st.write("Hasil Prediksi:")
                st.write(data_rf)

                # Dropdown untuk memilih informasi yang ingin ditampilkan
                info_type = st.selectbox('**Pilih informasi yang ingin ditampilkan:**', ['Metrik Evaluasi Model', 'Confusion Matrix'])

                if info_type == 'Metrik Evaluasi Model':
                    # Menampilkan metrik evaluasi
                    st.subheader('Metrik Evaluasi Model')

                    accuracy_rf = accuracy_score(true_labels_rf, predictions_rf)
                    auc_rf = roc_auc_score(true_labels_rf, predictions_rf)
                    tn, fp, fn, tp = confusion_matrix(true_labels_rf, predictions_rf).ravel()
                    specificity_rf = tn / (tn + fp)
                    sensitivity_rf = tp / (tp + fn)

                    st.write(f'**Akurasi**: {accuracy_rf:.2f}')
                    st.write(f'**Spesifisitas (Specificity)**: {specificity_rf:.2f}')
                    st.write(f'**Sensitivitas (Sensitivity)**: {sensitivity_rf:.2f}')
                    st.write(f'**AUC ROC**: {auc_rf:.2f}')

                elif info_type == 'Confusion Matrix':
                    # Menampilkan Confusion Matrix
                    st.subheader('Confusion Matrix')
                    cm_rf = confusion_matrix(true_labels_rf, predictions_rf)
                    plt.figure(figsize=(6, 4))
                    sns.heatmap(cm_rf, annot=True, cmap='Reds', fmt='g')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot()

                    # Menambahkan penjelasan di bawah Confusion Matrix
                    st.markdown(f"Transaksi yang sebenarnya **sah dan diprediksi sebagai sah** adalah sebesar {cm_rf[0, 0]} data")
                    st.markdown(f"Transaksi yang sebenarnya **sah tetapi diprediksi sebagai penipuan** adalah sebesar {cm_rf[0, 1]} data")
                    st.markdown(f"Transaksi yang sebenarnya **penipuan dan diprediksi sebagai penipuan** adalah sebesar {cm_rf[1, 1]} data")
                    st.markdown(f"Transaksi yang sebenarnya **penipuan tetapi diprediksi sebagai sah** adalah sebesar {cm_rf[1, 0]} data")

                # Mengkonversi DataFrame ke Excel menggunakan xlsxwriter
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
    Dashboard ini menggunakan algoritma ***Random Forest*** yang merupakan salah satu algoritma *machine learning* yang umum digunakan dalam permasalahan klasifikasi atau prediksi. Pada kasus ini, digunakan untuk memprediksi transaksi yang sah atau terindikasi sebagai penipuan berdasarkan jumlah transaksi dan jeda waktu transaksi (detik). Ilustrasinya adalah sebagai berikut:
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
    Terdapat beberapa pengukuran yang biasa digunakan untuk menentukan seberapa baik model, antara lain:
    
    - **Spesifisitas (Specificity)** mengukur kemampuan model untuk dengan benar mengidentifikasi negatif sejati (true negatives) di antara semua kasus yang sebenarnya negatif.
    - **Sensitivitas (Sensitivity)** mengukur kemampuan model untuk dengan benar mengidentifikasi positif sejati (true positives) di antara semua kasus yang sebenarnya positif.
    - **Akurasi (Accuracy)** mengukur seberapa sering model membuat prediksi yang benar, baik untuk kasus positif maupun negatif.
    - **AUC ROC (Area Under the Receiver Operating Characteristic Curve)** mengukur kinerja model klasifikasi pada berbagai threshold keputusan.
    - **ROC (Receiver Operating Characteristic Curve)** adalah grafik yang menggambarkan rasio True Positive Rate (Sensitivitas) terhadap False Positive Rate (1 - Spesifisitas) untuk berbagai nilai threshold.
    """)
