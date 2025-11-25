# Task Scheduling Cloud Computing Menggunakan Algoritma Cloudy-GSA

Proyek ini mendemonstrasikan implementasi algoritma penjadwalan tugas (_task scheduling_) pada lingkungan _cloud computing_ menggunakan **Cloudy Gravitational Search Algorithm (Cloudy-GSA)**. Algoritma ini bertujuan untuk mengoptimalkan alokasi serangkaian tugas (didefinisikan dalam `dataset.txt`) ke berbagai _Virtual Machines_ (VM) dengan spesifikasi yang berbeda. Tujuannya adalah untuk meminimalkan _makespan_ (waktu penyelesaian total) dan _degree of imbalance_ (tingkat ketidakseimbangan beban kerja) antar VM, sehingga pemanfaatan sumber daya menjadi lebih efisien.

## Penjelasan Algoritma: Cloudy-GSA

**Gravitational Search Algorithm (GSA)** adalah sebuah algoritma optimasi metaheuristik yang terinspirasi dari hukum gravitasi dan interaksi massa Newton. Dalam GSA, setiap solusi kandidat direpresentasikan sebagai sebuah "massa" atau "agen" dalam ruang pencarian. Kualitas dari setiap solusi (diukur dengan fungsi _fitness_) menentukan besarnya massa yang dimilikinya.

Prinsip kerja GSA adalah sebagai berikut:

1. Partikel. Setiap "partikel" merepresentasikan satu kemungkinan solusi penjadwalan. Misalnya, satu partikel adalah satu set pemetaan lengkap dari semua tugas ke semua VM.
2. Posisi. Posisi partikel dalam ruang pencarian menentukan VM mana yang ditugaskan untuk setiap tugas.
3. Massa. Setiap partikel memiliki "massa" yang dihitung dari seberapa bagus solusinya (nilai fitness). Semakin bagus solusinya (misalnya, makespan lebih rendah, beban lebih seimbang), semakin besar massanya.
4. Gaya Gravitasi. Partikel dengan massa yang lebih besar akan menarik partikel lain. Ini secara metaforis berarti solusi yang lebih baik akan "menarik" solusi lain untuk bergerak ke arah area yang lebih optimal di ruang pencarian.

**Cloudy-GSA**, seperti yang diimplementasikan dalam proyek ini, adalah varian dari GSA yang diadaptasi secara khusus untuk masalah penjadwalan tugas di lingkungan cloud. Terdapat beberapa modifikasi dan penyesuaian yang dilakukan untuk meningkatkan kinerja algoritma dalam konteks ini, seperti berikut:

1. **Unified Fitness Function.** Fungsi fitness tidak hanya mengukur makespan, tetapi merupakan gabungan dari tiga metrik dengan bobot tertentu:
   - Makespan: Waktu total yang dibutuhkan untuk menyelesaikan semua tugas. (Prioritas utama).
   - Load Imbalance: Penalti diberikan jika beban kerja tidak seimbang antar VM. Ini diukur dari standar deviasi beban VM.
   - Resource Utilization: Penalti diberikan jika utilisasi CPU secara keseluruhan rendah, mendorong agar VM tidak banyak menganggur.
2. **Intelligent Local Search.** Setelah setiap iterasi, beberapa solusi terbaik akan melalui proses pencarian lokal. Algoritma secara cerdas mencoba memindahkan tugas dari VM yang paling sibuk ke VM yang paling lengang untuk menemukan perbaikan kecil namun signifikan.
3. **Adaptive Inertia & Mutation.** Untuk menghindari terjebak di solusi optimal lokal, algoritma ini menggunakan:
   - Inersia Adaptif: "Kecepatan" partikel disesuaikan secara dinamis. Jika solusi membaik, partikel bergerak lebih lambat (presisi). Jika stagnan, partikel bergerak lebih cepat (eksplorasi).
   - Mutasi: Sejumlah kecil penugasan diubah secara acak untuk memperkenalkan keragaman dan menjelajahi area solusi baru.

# Alur Kerja

Alur kerja dari implementasi ini dapat dijelaskan dalam beberapa langkah utama:

1.  **Inisialisasi (`scheduler.py`):**
    - Memuat konfigurasi VM (IP, jumlah core CPU) dari file `.env`.
    - Membaca daftar tugas yang akan dieksekusi dari `dataset.txt`.
2.  **Penjadwalan (`cloudy_gsa_algorithm.py`):**
    - Fungsi `cloudy_gsa_scheduler` dijalankan sebanyak **10 kali (runs)** untuk memastikan stabilitas hasil.
    - Setiap run menjalankan algoritma GSA selama `1000` iterasi untuk mencari pemetaan terbaik.
3.  **Eksekusi Paralel (`scheduler.py`):**
    - Menggunakan `asyncio` dan `httpx`, scheduler mengirimkan permintaan tugas ke VM yang sesuai secara bersamaan.
    - **Semaphore** digunakan untuk setiap VM, membatasi jumlah tugas bersamaan sesuai dengan jumlah core CPU-nya.
4.  **Pengumpulan Hasil & Analisis (`scheduler.py`):**
    - Mencatat waktu mulai, selesai, dan durasi eksekusi.
    - Menyimpan hasil per-run ke file CSV terpisah.
    - Menghitung **rata-rata** metrik kinerja utama (Makespan, Throughput, dll) dari 10 kali pengujian.

## Cara Menjalankan

1.  Clone Repositori
    ```bash
    git clone https://github.com/Ax3lrod/SOKA-A-Kelompok-C-Task-Scheduling-Algorithm
    cd SOKA-A-Kelompok-C-Task-Scheduling-Algorithm
    ```
2.  Buat Virtual Environment
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```
3.  Install Dependensi
    ```bash
    pip install -r requirements.txt
    ```
4.  Konfigurasi VM
    - Buat file bernama `.env` seperti `.env.example`.
    - Isi dengan alamat IP dari setiap VM Anda.
5.  Siapkan Dataset
    - Buat file `dataset.txt` berisi daftar indeks beban tugas.
6.  Jalankan Scheduler
    ```bash
    python scheduler.py
    ```

## Hasil Eksekusi

Berikut adalah ringkasan hasil dari **10 kali pengujian (runs)** untuk memvalidasi performa algoritma secara statistik:

**Log di Konsol (Summary):**

```
...
Run 10: Mengeksekusi 20 tugas...
Run 10 selesai dalam 25.0267 detik.
Run 10 Metrics -> Makespan: 25.03s, Utilization: 57.60%

========================================
HASIL AKHIR RATA-RATA DARI 10 RUN
========================================
    Makespan  Throughput  Total CPU Time  ...  Avg Wait Time  Imbalance Degree  Resource Utilization
0  24.296705    0.823157      216.540288  ...       4.676608          0.966961              0.594155
1  23.936146    0.835556      194.803443  ...       5.872355          1.037584              0.542564
...
9  25.026683    0.799147      216.247394  ...       3.803095          1.202457              0.576045

----------------------------------------

RATA-RATA METRIK:
Makespan                 : 23.9815
Throughput               : 0.8345
Total CPU Time           : 211.2505
Total Wait Time          : 73.1094
Avg Execution Time       : 10.5625
Avg Wait Time            : 3.6555
Imbalance Degree         : 1.1052
Resource Utilization     : 58.7400%
```

## Pembahasan Hasil

Analisis dari hasil eksekusi (rata-rata dari 10 percobaan) menunjukkan efektivitas algoritma Cloudy-GSA yang dioptimalkan untuk penjadwalan tugas di lingkungan cloud computing:

1.  **Efisiensi Waktu (Makespan):** Rata-rata 20 tugas diselesaikan dalam **23.98 detik**. Jika dikerjakan secara serial, rata-rata total waktu komputasi (Total CPU Time) adalah **211.25 detik**. Ini menunjukkan bahwa penjadwalan dan eksekusi paralel berhasil mempercepat proses sekitar **8.8x lipat**. Konsistensi hasil antar run (sekitar 23-25 detik) juga menunjukkan algoritma cukup stabil.

2.  **Keseimbangan Beban (Imbalance Degree):** Nilai rata-rata **1.1052** menunjukkan adanya distribusi beban yang wajar di antara VM. Mengingat kapasitas VM sangat beragam (1, 2, 4, dan 8 core), nilai yang mendekati 1 ini menandakan algoritma berhasil mengenali kapasitas VM dan tidak membebani VM kecil secara berlebihan, meskipun keseimbangan sempurna (0) sulit dicapai pada lingkungan heterogen.

3.  **Utilisasi Sumber Daya (Resource Utilization):** Rata-rata utilisasi CPU sebesar **58.74%** adalah angka yang cukup baik untuk klaster heterogen. Ini berarti hampir 60% kapasitas komputasi total terus bekerja selama durasi _batch_. Penurunan sedikit dibanding tes sebelumnya wajar karena adanya variasi acak pada GSA dan total waktu CPU (beban tugas) pada pengujian ini sedikit lebih rendah (211s vs 240s).

4.  **Waktu Tunggu (Wait Time):** Rata-rata waktu tunggu per tugas adalah **3.66 detik**, dengan total waktu tunggu sistem **73.11 detik**. Meskipun terjadi antrian (terutama pada VM dengan core sedikit atau saat VM besar sedang penuh), durasi antrian ini masih relatif singkat dibandingkan rata-rata waktu eksekusi tugas (**10.56 detik**).

5.  **Trade-off Antara Imbalance dan Utilisasi:** Salah satu temuan menarik yang konsisten adalah strategi algoritma untuk meminimalkan makespan. Untuk menyelesaikan semua tugas secepat mungkin, algoritma cenderung membebani secara strategis VM yang paling kuat (seperti VM4 dengan 8 core). Akibatnya:
    - VM yang kuat ini menjadi "workhorse" yang terus-menerus sibuk.
    - VM yang lebih lemah mungkin menyelesaikan tugasnya lebih cepat dan menjadi idle, yang sedikit menaikkan _Imbalance Degree_.
    - Namun, strategi ini memastikan tugas-tugas berat diselesaikan oleh prosesor terkuat, yang pada akhirnya menekan _Makespan_ ke titik optimal (23.98 detik), meskipun harus mengorbankan sedikit kesetaraan beban.

Secara keseluruhan, hasil validasi 10 run ini membuktikan bahwa implementasi Cloudy-GSA mampu membuat keputusan penjadwalan yang **cerdas, stabil, dan efisien**, menyeimbangkan kecepatan penyelesaian dengan pemanfaatan sumber daya yang tersedia.
