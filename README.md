# Task Scheduling Cloud Computing Menggunakan Algoritma Cloudy-GSA

Proyek ini mendemonstrasikan implementasi algoritma penjadwalan tugas (*task scheduling*) pada lingkungan *cloud computing* menggunakan **Cloudy Gravitational Search Algorithm (Cloudy-GSA)**. Algoritma ini bertujuan untuk mengoptimalkan alokasi serangkaian tugas (didefinisikan dalam `dataset.txt`) ke berbagai *Virtual Machines* (VM) dengan spesifikasi yang berbeda. Tujuannya adalah untuk meminimalkan *makespan* (waktu penyelesaian total) dan *degree of imbalance* (tingkat ketidakseimbangan beban kerja) antar VM, sehingga pemanfaatan sumber daya menjadi lebih efisien.

## Penjelasan Algoritma: Cloudy-GSA
**Gravitational Search Algorithm (GSA)** adalah sebuah algoritma optimasi metaheuristik yang terinspirasi dari hukum gravitasi dan interaksi massa Newton. Dalam GSA, setiap solusi kandidat direpresentasikan sebagai sebuah "massa" atau "agen" dalam ruang pencarian. Kualitas dari setiap solusi (diukur dengan fungsi *fitness*) menentukan besarnya massa yang dimilikinya.

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

## Alur Kerja
Alur kerja dari implementasi ini dapat dijelaskan dalam beberapa langkah utama:
1.  **Inisialisasi (`scheduler.py`):**
    - Memuat konfigurasi VM (IP, jumlah core CPU) dari file `.env`.
    - Membaca daftar tugas yang akan dieksekusi dari `dataset.txt`. Setiap angka dalam dataset merepresentasikan "beban" komputasi dari sebuah tugas.
2.  **Penjadwalan (`cloudy_gsa_algorithm.py`):**
    - Fungsi `cloudy_gsa_scheduler` dipanggil.
    - Populasi solusi acak dibuat.
    - Algoritma GSA berjalan selama `1000` iterasi:
        - Mengevaluasi semua solusi menggunakan *Unified Fitness Function*.
        - Menghitung massa untuk setiap solusi.
        - Menghitung gaya, percepatan, dan kecepatan untuk setiap partikel (solusi).
        - Memperbarui posisi partikel (mengubah pemetaan tugas ke VM).
        - Menjalankan *Intelligent Local Search* dan *Mutation*.
    - Setelah selesai, algoritma mengembalikan pemetaan tugas-ke-VM terbaik yang ditemukan.
3.  **Eksekusi Paralel (`scheduler.py`):**
    - Menggunakan `asyncio` dan `httpx`, scheduler mulai mengirimkan permintaan tugas ke VM yang sesuai secara bersamaan.
    - **Semaphore** digunakan untuk setiap VM, yang membatasi jumlah tugas bersamaan yang dapat dijalankan pada satu VM sesuai dengan jumlah core CPU-nya. Ini mensimulasikan kemampuan pemrosesan paralel VM.
4.  **Pengumpulan Hasil & Analisis (`scheduler.py`):**
    - Setelah semua tugas selesai, skrip mencatat waktu mulai, waktu selesai, dan waktu eksekusi untuk setiap tugas.
    - Hasil mentah disimpan ke `result.csv`.
    - Metrik kinerja utama seperti **Makespan, Throughput, Imbalance Degree, dan Resource Utilization** dihitung dan ditampilkan di konsol.

## Cara Menjalankan
1.  Clone Repositori
    ```bash
    git clone <url-repositori>
    cd <nama-folder>
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
    - Buat file `dataset.txt`.
    - Isi dengan daftar indeks beban tugas (satu per baris). 
6.  Jalankan Scheduler
    ```bash
    python scheduler.py
    ```

## Hasil Eksekusi

**Log di Konsol:**
```
Berhasil memuat 20 tugas dari dataset.txt
Memulai Cloudy-GSA (V4 - Unified Fitness, Adaptive Inertia, 1000 iterasi)...
Estimasi Fitness Awal (Acak): 1463552.97
Cloudy-GSA Selesai. Fitness Terbaik: 528705.17

Penugasan Tugas Terbaik Ditemukan:
  - Tugas 0 -> vm3
  - ... (dan seterusnya)

Memulai eksekusi 20 tugas secara paralel...
Mengeksekusi task-6-0 (idx: 0) di vm3 (IP: 10.15.42.79)...
...
Selesai task-7-12 (idx: 12) di vm4. Waktu: 24.6664s

Semua eksekusi tugas selesai dalam 24.6841 detik.

Data hasil eksekusi disimpan ke result.csv

--- Hasil ---
Total Tugas Selesai       : 20
Makespan (Waktu Total)    : 24.6841 detik
Throughput                : 0.8102 tugas/detik
Total CPU Time            : 240.9826 detik
Total Wait Time           : 54.0254 detik
Average Start Time (rel)  : 2.7169 detik
Average Execution Time    : 12.0491 detik
Average Finish Time (rel) : 14.7660 detik
Imbalance Degree          : 1.1803
Resource Utilization (CPU): 65.0844%
```

**File `result.csv`:**
| index | task_name  | vm_assigned | start_time | exec_time             | finish_time | wait_time              |
|-------|------------|-------------|------------|------------------------|-------------|-------------------------|
| 0     | task-6-0   | vm3         | 0.0        | 9.937942000004114      | 9.937936    | 0.000011099997209385037 |
| 1     | task-5-1   | vm1         | 0.012528   | 10.713590899998962     | 10.726111   | 0.000009900002623908222 |
| 2     | task-8-2   | vm3         | 0.013751   | 18.33313509999425      | 18.34688    | 0.0000041000021155923605 |
| 3     | task-2-3   | vm2         | 0.014117   | 0.9817868000027374     | 0.995897    | 0.0000031999952625483274 |
| 4     | task-10-4  | vm4         | 0.014478   | 9.930038999998942      | 9.944512    | 0.0000027999994927085936 |
| 5     | task-3-5   | vm3         | 0.014949   | 10.935596099996474     | 10.95054    | 0.0000036000055843032897 |
| 6     | task-4-6   | vm4         | 0.015475   | 16.885121100000106     | 16.90059    | 0.0000036000055843032897 |
| 7     | task-4-7   | vm2         | 0.015984   | 4.495097699997132      | 4.511076    | 0.0000027999994927085936 |
| 8     | task-7-8   | vm3         | 0.016449   | 5.732756400000653      | 5.749201    | 0.0000034000040614046156 |
| 12    | task-7-12  | vm4         | 0.016957   | 24.66641559999698      | 24.683368   | 0.0000030000010156072676 |
| 13    | task-9-13  | vm4         | 0.017426   | 21.695626400003675     | 21.713046   | 0.0000037999998312443495 |
| 15    | task-8-15  | vm4         | 0.017883   | 3.98087790000136       | 3.998754    | 0.0000031000017770566046 |
| 19    | task-10-19 | vm4         | 0.018385   | 15.961220599994704     | 15.979599   | 0.0000026999987312592566 |
| 10    | task-9-10  | vm2         | 0.996626   | 21.02965740000218      | 22.026275   | 0.9795289999965462       |
| 14    | task-1-14  | vm2         | 4.511467   | 17.71485599999869      | 22.226317   | 4.493630700002541        |
| 9     | task-3-9   | vm3         | 5.749981   | 13.717444399997476     | 19.467421   | 5.733154099994863        |
| 16    | task-2-16  | vm3         | 9.943729   | 9.955731399997603      | 19.899455   | 9.925346600000921        |
| 11    | task-1-11  | vm1         | 10.726867  | 0.5420864999978221     | 11.268944   | 10.70976040000096        |
| 18    | task-6-18  | vm3         | 10.951111  | 13.016358600005333     | 23.967463   | 10.93274399999791        |
| 17    | task-5-17  | vm1         | 11.269613  | 10.757254299998749     | 22.026861   | 11.251172099997348       |


## Pembahasan Hasil

Analisis dari hasil eksekusi menunjukkan efektivitas algoritma Cloudy-GSA yang dioptimalkan untuk penjadwalan tugas di lingkungan cloud computing:

1.  **Efisiensi Waktu (Makespan):** Seluruh 20 tugas berhasil diselesaikan dalam **24.68 detik**. Jika dikerjakan secara serial, total waktu komputasi (Total CPU Time) adalah **240.98 detik**. Ini menunjukkan bahwa penjadwalan dan eksekusi paralel berhasil mengurangi waktu penyelesaian secara drastis (hampir 10x lebih cepat).

2.  **Keseimbangan Beban (Imbalance Degree):** Nilai `1.1803` menunjukkan adanya perbedaan beban kerja antara VM yang paling sibuk dan paling lengang. Nilai 0 menandakan keseimbangan sempurna. Angka ini wajar mengingat heterogenitas tugas dan kapasitas VM. Algoritma telah mencoba menyeimbangkan beban, tetapi tidak bisa sempurna karena sifat diskrit dari tugas-tugas tersebut.

3.  **Utilisasi Sumber Daya (Resource Utilization):** Utilisasi CPU sebesar **65.08%** adalah angka yang baik. Ini berarti selama proses eksekusi, lebih dari separuh kapasitas total CPU dari semua VM termanfaatkan secara efektif. Ini membuktikan bahwa *Unified Fitness Function* yang memberi penalti pada utilisasi rendah bekerja dengan baik.

4.  **Waktu Tunggu (Wait Time):** Total waktu tunggu sebesar **54.02 detik** menunjukkan bahwa antrian tugas terjadi. Ini disebabkan karena sebuah VM (misalnya, VM dengan 1 core) hanya bisa mengerjakan satu tugas pada satu waktu, sehingga tugas lain yang ditugaskan ke VM tersebut harus menunggu. Ini adalah trade-off yang tak terhindarkan dalam sistem penjadwalan.

5. **Trade-off Antara Imbalance dan Utilisasi:** Salah satu temuan menarik dari hasil ini adalah adanya potensi korelasi positif antara imbalance dan resource utilization. Meskipun terdengar kontradiktif, ini dapat dijelaskan oleh strategi algoritma untuk meminimalkan makespan. Untuk menyelesaikan semua tugas secepat mungkin, algoritma cenderung membebani secara strategis VM yang paling kuat (misalnya vm4 dengan 8 core). Akibatnya:
    - VM yang kuat ini menjadi "workhorse" yang terus-menerus sibuk, memaksimalkan utilisasi core-nya selama durasi eksekusi.
    - VM yang lebih lemah mungkin menyelesaikan tugasnya lebih cepat dan menjadi idle, sehingga menciptakan ketidakseimbangan (imbalance) beban yang lebih tinggi.
    - Namun, karena sumber daya komputasi terbesar terus bekerja secara efisien, utilisasi sumber daya gabungan (overall resource utilization) cenderung meningkat. Dengan kata lain, algoritma ini rela mengorbankan kesetaraan beban demi memastikan sumber daya yang paling kapabel tidak pernah menganggur, yang pada akhirnya menekan waktu penyelesaian total.

Secara keseluruhan, hasil ini memvalidasi bahwa implementasi Cloudy-GSA ini mampu membuat keputusan penjadwalan yang cerdas, menghasilkan eksekusi yang efisien, dan menyeimbangkan berbagai tujuan (kecepatan, keseimbangan, dan utilisasi) secara efektif. 