import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
import io
import xlsxwriter

# Fungsi untuk memuat model (dummy untuk simulasi)
def load_model():
    return None

# Memuat model
trans_model = load_model()

# Fungsi untuk memuat data transaksi dari file Excel
def load_data(file):
    data = pd.read_excel(file)
    return data

# Fungsi untuk melakukan prediksi
def predict(model, data):
    predictions = model.predict(data)
    return predictions

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Prediksi Transaksi",
    layout="wide",
    page_icon="💱"
)

# Sidebar untuk navigasi
with st.sidebar:
    selected = st.selectbox(
        "Navigasi",
        [
            "Manual Input",
            "**Upload file excel yang berisi data TX_AMOUNT dan TX_TIME_SECONDS**",
            "**Upload file Excel yang berisi data TX_AMOUNT, TX_TIME_SECONDS dan FRAUD**",
            "Info"
        ],
        index=0,
    )

# Halaman input manual
if selected == "Manual Input":
    st.title("Transaction Prediction - Manual Input")

    col1, col2 = st.columns(2)

    with col1:
        TX_AMOUNT = st.text_input("Jumlah Transaksi", "")
    with col2:
        TX_TIME_SECONDS = st.text_input("Jeda Waktu Transaksi (Detik)", "")

    transaction_prediction = ""

    if st.button("Transaction Prediction Result"):
        try:
            user_input = pd.DataFrame({
                "TX_AMOUNT": [float(TX_AMOUNT)],
                "TX_TIME_SECONDS": [float(TX_TIME_SECONDS)],
            })
            prediction = predict(trans_model, user_input)
            if prediction[0] == 1:
                transaction_prediction = "Transaksi tidak aman karena terjadi indikasi penipuan"
            else:
                transaction_prediction = "Transaksi aman karena dilakukan secara sah"
        except ValueError:
            transaction_prediction = "Harap masukkan nilai numerik yang valid untuk semua input"

        st.success(transaction_prediction)

# Halaman upload file (untuk transaksi biasa)
elif selected == "**Upload file excel yang berisi data TX_AMOUNT dan TX_TIME_SECONDS**":
    st.title("Transaction Prediction - File Upload")

    uploaded_file = st.file_uploader(
        "**Upload file excel yang berisi data TX_AMOUNT dan TX_TIME_SECONDS**", type=["xlsx"]
    )

    if uploaded_file is not None:
        try:
            data = load_data(uploaded_file)
            st.write("Data yang diupload untuk prediksi transaksi biasa:")
            st.write(data)

            if "TX_AMOUNT" in data.columns and "TX_TIME_SECONDS" in data.columns:
                # Menampilkan tabel statistik deskriptif
                st.subheader("Karakteristik Jeda Waktu Detik")
                st.write(
                    data["TX_TIME_SECONDS"]
                    .describe()
                    .to_frame()
                    .T[["mean", "50%", "std"]]
                    .rename(
                        columns={"mean": "Rata-Rata", "50%": "Median", "std": "Varians"}
                    )
                )

                st.subheader("Karakteristik Jumlah Transaksi")
                st.write(
                    data["TX_AMOUNT"]
                    .describe()
                    .to_frame()
                    .T[["mean", "50%", "std"]]
                    .rename(
                        columns={"mean": "Rata-Rata", "50%": "Median", "std": "Varians"}
                    )
                )

                # Dropdown untuk memilih tipe plot
                plot_type = st.selectbox("Pilih jenis plot:", ["Histogram", "Boxplot"])

                if plot_type == "Histogram":
                    # Menampilkan histogram
                    st.subheader("Distribusi Jumlah Transaksi")
                    fig_hist, ax_hist = plt.subplots()
                    sns.histplot(data["TX_AMOUNT"], kde=True, ax=ax_hist, color="lightcoral")
                    ax_hist.set_xlabel("TX_AMOUNT")
                    ax_hist.set_ylabel("Frekuensi")
                    st.pyplot(fig_hist)
                    st.write(
                        "Histogram yang berbentuk melengkung seperti lonceng menggambarkan bahwa data berdistribusi normal dan selain itu berbentuk tidak normal."
                    )

                    st.subheader("Distribusi Jeda Waktu Transaksi (Detik)")
                    fig_hist_time, ax_hist_time = plt.subplots()
                    sns.histplot(
                        data["TX_TIME_SECONDS"], kde=True, ax=ax_hist_time, color="lightcoral"
                    )
                    ax_hist_time.set_xlabel("TX_TIME_SECONDS")
                    ax_hist_time.set_ylabel("Frekuensi")
                    st.pyplot(fig_hist_time)
                    st.write(
                        "Histogram yang berbentuk melengkung seperti lonceng menggambarkan bahwa data berdistribusi normal dan selain itu berbentuk tidak normal."
                    )

                elif plot_type == "Boxplot":
                    # Menampilkan boxplot
                    st.subheader("Boxplot Jumlah Transaksi")
                    fig_box, ax_box = plt.subplots()
                    sns.boxplot(data["TX_AMOUNT"], ax=ax_box, color="lightcoral")
                    ax_box.set_xlabel("TX_AMOUNT")
                    st.pyplot(fig_box)
                    st.write(
                        "Boxplot yang berbentuk lebar menandakan bahwa penyebaran datanya tinggi dan sebaliknya apabila berbentuk sempit menandakan bahwa penyebaran data rendah. Titik di luar kotak boxplot merupakan data outlier yang nilainya berbeda jauh dari nilai lainnya pada data."
                    )

                    st.subheader("Boxplot Jeda Waktu Transaksi (Detik)")
                    fig_box_time, ax_box_time = plt.subplots()
                    sns.boxplot(data["TX_TIME_SECONDS"], ax=ax_box_time, color="lightcoral")
                    ax_box_time.set_xlabel("TX_TIME_SECONDS")
                    st.pyplot(fig_box_time)
                    st.write(
                        "Boxplot yang berbentuk lebar menandakan bahwa penyebaran datanya tinggi dan sebaliknya apabila berbentuk sempit menandakan bahwa penyebaran data rendah. Titik di luar kotak boxplot merupakan data outlier yang nilainya berbeda jauh dari nilai lainnya pada data."
                    )

                # Mengkonversi DataFrame ke Excel menggunakan xlsxwriter tanpa engine_kwargs
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    data.to_excel(writer, index=False, sheet_name="Sheet1")

                output.seek(0)

                st.download_button(
                    label="Download hasil prediksi",
                    data=output,
                    file_name="hasil_prediksi.xlsx",
                )

            else:
                st.error("File tidak memiliki kolom yang diperlukan: TX_AMOUNT, TX_TIME_SECONDS")
        except Exception as e:
            st.error(f"Error: {e}")

# Halaman informasi
elif selected == "**Upload file Excel yang berisi data TX_AMOUNT, TX_TIME_SECONDS dan FRAUD**":
    st.title("Transaction Prediction - File Upload (dengan FRAUD)")

    uploaded_file = st.file_uploader(
        "**Upload file Excel yang berisi data TX_AMOUNT, TX_TIME_SECONDS dan FRAUD**", type=["xlsx"]
    )

    if uploaded_file is not None:
        try:
            data = load_data(uploaded_file)
            st.write("Data yang diupload untuk prediksi dengan FRAUD:")
            st.write(data)

            if "TX_AMOUNT" in data.columns and "TX_TIME_SECONDS" in data.columns and "FRAUD" in data.columns:
                # Menampilkan tabel statistik deskriptif
                st.subheader("Karakteristik Jeda Waktu Detik")
                st.write(
                    data["TX_TIME_SECONDS"]
                    .describe()
                    .to_frame()
                    .T[["mean", "50%", "std"]]
                    .rename(
                        columns={"mean": "Rata-Rata", "50%": "Median", "std": "Varians"}
                    )
                )

                st.subheader("Karakteristik Jumlah Transaksi")
                st.write(
                    data["TX_AMOUNT"]
                    .describe()
                    .to_frame()
                    .T[["mean", "50%", "std"]]
                    .rename(
                        columns={"mean": "Rata-Rata", "50%": "Median", "std": "Varians"}
                    )
                )

                # Dropdown untuk memilih tipe plot
                plot_type = st.selectbox("Pilih jenis plot:", ["Histogram", "Boxplot"])

                if plot_type == "Histogram":
                    # Menampilkan histogram
                    st.subheader("Distribusi Jumlah Transaksi")
                    fig_hist, ax_hist = plt.subplots()
                    sns.histplot(data["TX_AMOUNT"], kde=True, ax=ax_hist, color="lightcoral")
                    ax_hist.set_xlabel("TX_AMOUNT")
                    ax_hist.set_ylabel("Frekuensi")
                    st.pyplot(fig_hist)
                    st.write(
                        "Histogram yang berbentuk melengkung seperti lonceng menggambarkan bahwa data berdistribusi normal dan selain itu berbentuk tidak normal."
                    )

                    st.subheader("Distribusi Jeda Waktu Transaksi (Detik)")
                    fig_hist_time, ax_hist_time = plt.subplots()
                    sns.histplot(
                        data["TX_TIME_SECONDS"], kde=True, ax=ax_hist_time, color="lightcoral"
                    )
                    ax_hist_time.set_xlabel("TX_TIME_SECONDS")
                    ax_hist_time.set_ylabel("Frekuensi")
                    st.pyplot(fig_hist_time)
                    st.write(
                        "Histogram yang berbentuk melengkung seperti lonceng menggambarkan bahwa data berdistribusi normal dan selain itu berbentuk tidak normal."
                    )

                elif plot_type == "Boxplot":
                    # Menampilkan boxplot
                    st.subheader("Boxplot Jumlah Transaksi")
                    fig_box, ax_box = plt.subplots()
                    sns.boxplot(data["TX_AMOUNT"], ax=ax_box, color="lightcoral")
                    ax_box.set_xlabel("TX_AMOUNT")
                    st.pyplot(fig_box)
                    st.write(
                        "Boxplot yang berbentuk lebar menandakan bahwa penyebaran datanya tinggi dan sebaliknya apabila berbentuk sempit menandakan bahwa penyebaran data rendah. Titik di luar kotak boxplot merupakan data outlier yang nilainya berbeda jauh dari nilai lainnya pada data."
                    )

                    st.subheader("Boxplot Jeda Waktu Transaksi (Detik)")
                    fig_box_time, ax_box_time = plt.subplots()
                    sns.boxplot(data["TX_TIME_SECONDS"], ax=ax_box_time, color="lightcoral")
                    ax_box_time.set_xlabel("TX_TIME_SECONDS")
                    st.pyplot(fig_box_time)
                    st.write(
                        "Boxplot yang berbentuk lebar menandakan bahwa penyebaran datanya tinggi dan sebaliknya apabila berbentuk sempit menandakan bahwa penyebaran data rendah. Titik di luar kotak boxplot merupakan data outlier yang nilainya berbeda jauh dari nilai lainnya pada data."
                    )

                # Mengkonversi DataFrame ke Excel menggunakan xlsxwriter tanpa engine_kwargs
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    data.to_excel(writer, index=False, sheet_name="Sheet1")

                output.seek(0)

                st.download_button(
                    label="Download hasil prediksi",
                    data=output,
                    file_name="hasil_prediksi.xlsx",
                )

            else:
                st.error(
                    "File tidak memiliki kolom yang diperlukan: TX_AMOUNT, TX_TIME_SECONDS, dan FRAUD"
                )
        except Exception as e:
            st.error(f"Error: {e}")

# Halaman informasi
elif selected == "Info":
    st.title("Informasi Dashboard")

    st.write(
        """
    *Random Forest* adalah salah satu algoritma machine learning yang umum digunakan dalam permasalahan klasifikasi atau prediksi. Pada kasus ini, digunakan untuk memprediksi mana transaksi yang termasuk ke dalam kelas penipuan dan sah. Prediksi didasarkan pada jumlah transaksi dan jeda waktu transaksi (detik).
    """
    )

    # Menampilkan gambar Random Forest dengan st.image dan mengatur penempatan dengan CSS
    st.markdown(
        """
    <style>
    .center {
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-width: 100%;
        height: auto;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<img src="https://cdn.prod.website-files.com/61af164800e38cf1b6c60b55/64c0c20d61bda9e68f630468_Random%20forest.webp" alt="Random Forest" width="400" class="center">',
        unsafe_allow_html=True,
    )

    st.write(
        """
    Terdapat beberapa pengukuran yang biasa digunakan untuk menentukan seberapa baik model, antara lain:
    
    - *Spesifisitas (Specificity)* mengukur kemampuan model untuk dengan benar mengidentifikasi negatif sejati (true negatives) di antara semua kasus yang sebenarnya negatif.
    - *Sensitivitas (Sensitivity)* mengukur kemampuan model untuk dengan benar mengidentifikasi positif sejati (true positives) di antara semua kasus yang sebenarnya positif.
    - *Akurasi (Accuracy)* mengukur seberapa sering model membuat prediksi yang benar, baik untuk kasus positif maupun negatif.
    - *AUC ROC (Area Under the Receiver Operating Characteristic Curve)* mengukur kinerja model klasifikasi pada berbagai threshold keputusan.
    - *ROC (Receiver Operating Characteristic Curve)* adalah grafik yang menggambarkan rasio True Positive Rate (Sensitivitas) terhadap False Positive Rate (1 - Spesifisitas) untuk berbagai nilai threshold.
    """
    )
