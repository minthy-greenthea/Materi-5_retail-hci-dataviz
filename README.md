# RetailSense: Chronicle of Customer Intelligence & AI Segmentation Dashboard

> **Tugas Studi Kasus & Implementasi Kuliah Umum Praktisi Informatika**  
> **Program:** _Informatics Practitioner Lecture Series: Connecting Academic Knowledge with Industry and Global Technology_  
> **Topik Materi:** _Human-Computer Interaction & Data Visualization — "Make Sense of Data with Analytics and AI"_  
> **Pemateri:** Bapak Arafat Febriandirza  
> **Penyelenggara:** Program Studi Informatika, Fakultas Teknik, Universitas Bengkulu

---

## 1. Latar Belakang & Rumusan Masalah Studi Kasus

Dalam industri ritel modern, transaksi harian menghasilkan data pelanggan dalam jumlah besar (_raw transaction logs_). Namun, manajemen dan pengambil keputusan sering kali mengalami kesulitan (_information overload_) ketika harus menganalisis ribuan baris tabel angka secara manual untuk memahami perilaku pelanggan.

Tanpa adanya integrasi antara **Analytics & AI**, **Data Visualization**, dan **Human-Computer Interaction (HCI)**, perusahaan ritel menghadapi tiga masalah utama:

1. **Kesulitan Memahami Pola Data (_Making Sense of Data_):** Data mentah seperti riwayat hari belanja terakhir, frekuensi transaksi, dan nominal pengeluaran sulit diterjemahkan secara langsung menjadi strategi bisnis.
2. **Tingginya Beban Kognitif (_High Cognitive Load_):** Tabel statis yang padat membuat pengguna cepat lelah dan rawan melewatkan anomali penting, seperti pelanggan yang hampir berhenti berbelanja (_churn / at-risk_).
3. **Kurangnya Rekomendasi Berbasis Data (_Lack of Actionable Insights_):** Manajemen membutuhkan sistem yang tidak hanya menampilkan data masa lalu (_Descriptive_), tetapi juga mampu mengelompokkan pelanggan secara otomatis (_Predictive_) dan memberikan saran keputusan (_Prescriptive_).

Sebagai solusi atas permasalahan tersebut, dibangunlah **RetailSense**, sebuah _interactive intelligence dashboard_ yang menggabungkan desain antarmuka ergonomis, visualisasi data multidimensi, dan algoritma _Machine Learning_ (**K-Means Clustering**).

---

## 2. Keterkaitan Implementasi dengan Materi Kuliah Umum

Proyek ini dirancang secara khusus dengan menerapkan **tiga pilar utama** yang dipaparkan dalam kuliah umum:

### A. Penerapan Prinsip _Human-Computer Interaction_ (HCI)

- **Visual Ergonomics & Aesthetic-Usability:** Antarmuka dirancang menggunakan palet warna _Sage Green_ (`#9CAF88`), _Warm Alabaster Cream_ (`#F4F3EB`), dan aksen _Champagne Gold_ (`#C5A869`) dengan tingkat kontras teks yang tajam (`#1A241B`). Pemilihan warna bernuansa alami yang tenang ini bertujuan mengurangi kelelahan mata (_visual fatigue_) pengguna saat berinteraksi dengan data dalam durasi lama.
- **Reduksi Beban Kognitif (_Cognitive Load Reduction_):** Informasi tidak ditumpuk dalam satu layar penuh, melainkan diorganisasikan secara bertahap menggunakan **4 Kartu Metrik Eksekutif (KPI Cards)** di bagian atas dan **3 Navigasi Tab** terpisah sesuai konteks analisis.
- **Prinsip _Visual Information-Seeking Mantra_ (Ben Shneiderman):**
  1. _Overview First:_ Menampilkan ringkasan total pelanggan aktif, akumulasi nilai transaksi (Rp 2,54 Miliar), indeks kepuasan pelanggan, dan jumlah segmen pelanggan yang memerlukan retensi di bagian atas layar.
  2. _Zoom and Filter:_ Menyediakan panel kontrol interaktif di bagian _sidebar_ kiri untuk menyaring data berdasarkan Kota, Kategori Produk, dan Rentang Usia secara _real-time_.
  3. _Details on Demand:_ Menyediakan _hover tooltip_ pada setiap titik grafik serta tabel arsip data lengkap pada bagian bawah _Tab 3_.

### B. Strategi _Data Visualization_ (_"Making Sense of Data"_)

Setiap jenis grafik dipilih berdasarkan fungsi komunikasinya terhadap persepsi visual manusia:

- **2D Bubble Scatter Plot:** Memetakan korelasi antara _Frequency_ (frekuensi transaksi) dan _Monetary_ (total nilai belanja), dengan ukuran titik merepresentasikan skor kepuasan dan warna merepresentasikan segmen hasil klasterisasi AI.
- **Donut Chart:** Menampilkan komposisi dan proporsi persentase dari keempat segmen pelanggan secara proporsional.
- **Grouped Bar Chart:** Membandingkan kontribusi nilai transaksi antar kategori produk di setiap segmen pelanggan.
- **Interactive 3D Scatter Plot & Box Plot:** Memungkinkan pengguna mengeksplorasi ruang 3 dimensi **RFM (_Recency, Frequency, Monetary_)** secara bebas (diputar dan diperbesar) serta mendiagnosis distribusi kepuasan pelanggan di tiap klaster.

### C. Integrasi _Analytics & Artificial Intelligence_ (AI)

Sistem ini menerapkan analisis perilaku pelanggan menggunakan metode **RFM Analysis** yang dipadukan dengan algoritma _Unsupervised Machine Learning_ **K-Means Clustering** (`scikit-learn`):

1. **Standardisasi Fitur:** Menyamakan skala variabel `Recency_Days`, `Frequency_Orders`, dan `Monetary_Spend` menggunakan `StandardScaler`.
2. **Klasterisasi Otomatis (K = 4):** Algoritma AI membagi 500 pelanggan ke dalam 4 segmen perilaku:
   - **Champions (Pelanggan Utama):** Transaksi baru, frekuensi sangat tinggi, dan nilai belanja terbesar.
   - **Loyal (Pelanggan Setia):** Berbelanja secara rutin dengan nilai transaksi menengah ke atas.
   - **Potential (Pelanggan Potensial):** Pelanggan baru atau berfrekuensi rendah yang potensial untuk ditingkatkan.
   - **At-Risk (Perlu Retensi):** Pelanggan yang sudah lama tidak bertransaksi dan memerlukan tindak lanjut retensi.
3. **Interactive AI Simulator (_Prescriptive Analytics_):** Pada _Tab 3_, pengguna dapat menggeser _slider_ parameter pelanggan secara langsung untuk melihat bagaimana model AI mengklasifikasikan pelanggan tersebut beserta rekomendasi tindakan bisnis yang harus diambil.

---

## 3. Arsitektur & Struktur File Proyek

    retail-hci-dataviz/
    ├── app.py                  # Source code utama Dashboard HCI, Data Visualization & AI Simulator
    ├── generate_data.py        # Script generator dataset sintetis 500 pelanggan retail
    ├── retail_customers.csv    # Dataset transaksi dan profil pelanggan (RFM & Satisfaction)
    ├── requirements.txt        # Daftar pustaka (dependencies) Python
    └── README.md               # Dokumentasi laporan studi kasus dan panduan penggunaan

---

## 4. Panduan Instalasi & Cara Menjalankan Aplikasi

Ikuti langkah-langkah berikut untuk menjalankan aplikasi dashboard di komputer lokal:

1. **Clone repositori ini ke komputer lokal:**

   git clone [https://github.com/USERNAME_ANDA/retail-hci-dataviz.git](https://github.com/USERNAME_ANDA/retail-hci-dataviz.git)
   cd retail-hci-dataviz

2. **Instal seluruh pustaka Python yang dibutuhkan:**

   pip install -r requirements.txt

3. **(Opsional) Bangkitkan ulang dataset pelanggan melalui generator:**

   python generate_data.py

4. **Jalankan aplikasi dashboard interaktif Streamlit:**

   python -m streamlit run app.py

5. **Akses di Browser:**  
   Aplikasi akan terbuka secara otomatis di peramban (_browser_) pada alamat `http://localhost:8501`.

---

## 5. Kesimpulan Studi Kasus

Melalui implementasi **RetailSense**, terbukti bahwa sinergi antara **Analytics & AI**, **Data Visualization**, dan **Human-Computer Interaction (HCI)** mampu mentransformasi 500 baris data transaksi mentah menjadi informasi visual yang intuitif dan bernilai strategis. Algoritma _K-Means Clustering_ bekerja mengekstraksi pola perilaku pelanggan secara otomatis, visualisasi interaktif menerjemahkan hasil komputasi tersebut ke dalam grafik yang mudah dipahami, dan desain antarmuka berpusat pada pengguna memastikan proses eksplorasi data hingga pengambilan keputusan berlangsung secara efisien tanpa membebani kognisi manusia.
