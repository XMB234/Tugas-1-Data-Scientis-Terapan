# Business Understanding
## HR Analytics — Employee Attrition
### PT Jaya Jaya Maju | Data Science Project 2025

## 1. Latar Belakang Bisnis

PT Jaya Jaya Maju merupakan perusahaan multinasional yang bergerak di bidang industri manufaktur dan layanan korporasi. Perusahaan ini berdiri sejak tahun 2000 dan telah berkembang menjadi salah satu pemain besar di industri nasional dengan lebih dari 1.000 karyawan yang tersebar di berbagai wilayah di Indonesia.

Selama lebih dari dua dekade beroperasi, Jaya Jaya Maju telah membangun reputasi yang solid dalam hal kualitas produk dan layanan. Namun, di balik pencapaian tersebut, perusahaan menghadapi tantangan internal yang cukup serius di sisi pengelolaan sumber daya manusia (SDM). Departemen HR mengidentifikasi bahwa tingkat pergantian karyawan (*employee attrition*) terus meningkat dari tahun ke tahun dan mulai berdampak pada operasional dan kinerja organisasi secara keseluruhan.

Berdasarkan data internal yang tersedia, attrition rate aktual perusahaan tercatat sebesar **16,9%** — jauh melampaui ambang batas toleransi yang ditetapkan perusahaan sebesar 10%. Kondisi ini mengindikasikan adanya masalah struktural yang perlu diidentifikasi dan ditangani secara sistematis menggunakan pendekatan berbasis data.

### 1.1 Konteks Industri

Tingginya attrition rate bukan hanya masalah internal, melainkan juga mencerminkan tekanan kompetitif di pasar tenaga kerja. Dalam industri manufaktur dan korporasi modern, perusahaan bersaing ketat untuk mempertahankan talenta terbaik. Biaya penggantian satu karyawan diperkirakan setara dengan **50–200% dari gaji tahunan** karyawan tersebut, mencakup biaya rekrutmen, pelatihan, dan hilangnya produktivitas selama masa transisi.

### 1.2 Profil Perusahaan

| Atribut | Detail |
|---|---|
| Nama Perusahaan | PT Jaya Jaya Maju |
| Tahun Berdiri | 2000 |
| Jumlah Karyawan | 1.470+ karyawan |
| Departemen | Sales, Research & Development, Human Resources |
| Skala Operasi | Multinasional — tersebar di seluruh Indonesia |
| Tantangan Utama | Attrition rate 16,9% — melebihi target maksimal 10% |

---

## 2. Permasalahan Bisnis

Tingginya tingkat attrition di Jaya Jaya Maju menimbulkan berbagai dampak negatif yang saling berkaitan, baik dari sisi operasional, finansial, maupun budaya organisasi.

### 2.1 Permasalahan Utama

1. **Attrition melampaui target:** Tingkat attrition mencapai 16,9%, jauh melampaui batas toleransi 10% yang ditetapkan oleh manajemen.
2. **Tidak ada early warning system:** Perusahaan belum memiliki sistem prediksi dini yang mampu mengidentifikasi karyawan berisiko tinggi untuk resign sebelum keputusan tersebut diambil.
3. **Minimnya alat monitoring:** Manajer HR kesulitan memonitori berbagai faktor penyebab attrition secara real-time karena tidak adanya dashboard terpusat.
4. **Keputusan tidak berbasis data:** Tanpa analisis berbasis data, keputusan HR dibuat berdasarkan intuisi dan pengalaman subjektif, bukan bukti empiris.

### 2.2 Dampak Bisnis

Tingginya attrition rate secara langsung berdampak pada empat aspek kritis perusahaan:

- **Finansial:** Biaya rekrutmen dan pelatihan karyawan baru meningkat signifikan setiap tahunnya.
- **Produktivitas:** Kehilangan karyawan berpengalaman mengganggu proyek yang sedang berjalan dan menurunkan output tim.
- **Budaya organisasi:** Perpindahan karyawan yang sering menciptakan ketidakpastian di dalam tim dan melemahkan kohesi organisasi.
- **Employer branding:** Reputasi perusahaan sebagai tempat kerja yang baik dapat tergerus jika masalah ini tidak segera diatasi.

### 2.3 Pertanyaan Analitik (*Business Questions*)

Proyek ini diarahkan untuk menjawab pertanyaan-pertanyaan analitik berikut:

1. Faktor-faktor apa saja yang paling signifikan mempengaruhi keputusan karyawan untuk resign?
2. Apakah ada pola tertentu pada karyawan yang resign berdasarkan departemen, usia, tingkat gaji, atau lama bekerja?
3. Karyawan dengan profil seperti apa yang memiliki risiko attrition paling tinggi?
4. Seberapa akurat model machine learning dapat memprediksi karyawan yang akan resign?
5. Intervensi HR apa yang paling efektif untuk menurunkan attrition rate berdasarkan temuan data?

---

## 3. Cakupan Proyek

Proyek ini dirancang sebagai proyek *end-to-end data science* yang mencakup seluruh tahapan dari pemahaman data hingga menghasilkan rekomendasi bisnis yang *actionable*.

| No. | Komponen | Deskripsi | Keterangan |
|---|---|---|---|
| 1 | Data Understanding | Eksplorasi awal dataset: shape, tipe data, distribusi, missing values | Selesai di awal proyek |
| 2 | Data Preparation | Handling missing values, encoding, feature selection, train-test split | Google Colab |
| 3 | Modeling | Membangun model prediksi attrition (klasifikasi biner: 0 = stay, 1 = resign) | Google Colab |
| 5 | Evaluation | Mengukur performa model: Accuracy, Precision, Recall, F1-Score, ROC-AUC | Google Colab |
| 6 | Business Dashboard | Dashboard interaktif untuk monitoring faktor attrition secara real-time | Metabase / Tableau |
| 7 | Rekomendasi Bisnis | Actionable insight berbasis hasil model dan EDA untuk tim HR | Laporan akhir |

---

## 4. Persiapan
### 4.1 Sumber Data

| Atribut | Detail |
|---|---|
| Nama Dataset | Employee Data — PT Jaya Jaya Maju |
| Sumber | Dicoding Academy Dataset (GitHub) |
| URL | [employee_data.csv](https://github.com/dicodingacademy/dicoding_dataset/blob/main/employee/employee_data.csv) |
| Jumlah Baris | 1.470 |
| Jumlah Kolom | 35 |
| Target Variabel | `Attrition` (0 = Stay, 1 = Resign) |
| Missing Values | 412 baris pada kolom `Attrition` |
| Lisensi | Dicoding Academy |

```python
DATASET_URL = (
    "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset"
    "/bce7a57a496d083716138922bc5839b5c30fa4ea/employee/employee_data.csv"
)
```

---

### 4.2 Setup Environment

#### 4.2.1 Prasyarat

| Tools | Kegunaan |
|---|---|
| Python | Bahasa pemrograman utama |
| Google Colab / Jupyter | Environment notebook |
| Docker Desktop | Menjalankan Metabase |
| Metabase | Business Dashboard |

---

#### 4.2.2 Instalasi Library

Buat file `requirements.txt` dengan isi berikut:

```
imbalanced-learn==0.14.1
lightgbm==4.6.0
matplotlib==3.10.0
missingno==0.5.2
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
scikit-plot==0.3.7
seaborn==0.13.2
shap==0.51.0
xgboost==3.2.0
```

Install semua library sekaligus:

```bash
pip install -r requirements.txt
```

Atau install satu per satu di Google Colab:

```python
!pip install -q imbalanced-learn==0.14.1
!pip install -q lightgbm==4.6.0
!pip install -q matplotlib==3.10.0
!pip install -q missingno==0.5.2
!pip install -q numpy==2.0.2
!pip install -q pandas==2.2.2
!pip install -q scikit-learn==1.6.1
!pip install -q scikit-plot==0.3.7
!pip install -q seaborn==0.13.2
!pip install -q shap==0.51.0
!pip install -q xgboost==3.2.0
```

---

#### 4.2.3 Deskripsi Library

| Library | Versi | Kegunaan |
|---|---|---|
| `pandas` | 2.2.2 | Manipulasi dan analisis data tabular |
| `numpy` | 2.0.2 | Operasi array dan komputasi numerik |
| `matplotlib` | 3.10.0 | Visualisasi data dasar |
| `seaborn` | 0.13.2 | Visualisasi statistik berbasis matplotlib |
| `missingno` | 0.5.2 | Visualisasi missing values |
| `scikit-learn` | 1.6.1 | Preprocessing, modeling, dan evaluasi ML |
| `imbalanced-learn` | 0.14.1 | Penanganan imbalanced dataset (SMOTE) |
| `xgboost` | 3.2.0 | Model XGBoost classifier |
| `lightgbm` | 4.6.0 | Model LightGBM classifier |
| `shap` | 0.51.0 | Model explainability (feature importance) |
| `scikit-plot` | 0.3.7 | Visualisasi ROC curve dan confusion matrix |

---

#### 4.2.4 Setup Metabase (Business Dashboard)

```bash
# Pull image Metabase v0.46.4
docker pull metabase/metabase:v0.46.4

# Jalankan container
docker run -p 3000:3000 --name metabase metabase/metabase:v0.46.4

# Akses via browser
# http://localhost:3000
# Email    : root@mail.com
# Password : root123
```
---

### 4.3 Panduan Penggunaan Inference Script
### Prediksi Attrition pada Data Karyawan Baru

---

#### 4.3.1 Gambaran Umum

Inference script ini digunakan untuk **memprediksi risiko attrition karyawan baru** menggunakan model XGBoost yang telah dilatih sebelumnya. Script memuat model dan preprocessing artifacts yang tersimpan, lalu menerapkannya pada data baru tanpa perlu melatih ulang model.

```
Data Karyawan Baru (CSV)
        │
        ▼
[1] Muat Model & Artifacts   → model_artifacts.pkl
        │
        ▼
[2] Preprocessing            → Encoding + Scaling
        │
        ▼
[3] Prediksi                 → Attrition_Predicted + Resign_Probability
        │
        ▼
[4] Output                   → Risk Level (Low / Medium / High)
```

---

#### 4.3.2 Prasyarat

#### 4.3.2.1 File yang Dibutuhkan

Pastikan file berikut tersedia di direktori yang sama dengan notebook:

| File | Keterangan |
|---|---|
| `model_artifacts.pkl` | Model + preprocessing artifacts (hasil dari tahap modeling) |
| `new_employee_data.csv` | Data karyawan baru yang ingin diprediksi *(opsional)* |

#### 4.3.2.2 Struktur `model_artifacts.pkl`

File ini berisi semua komponen yang diperlukan untuk inference:

```python
model_artifacts = {
    'best_model'                  : # Model XGBoost terlatih
    'best_model_name'             : # Nama model ('XGBoost')
    'scaler'                      : # StandardScaler (fit pada training data)
    'le_dict'                     : # Label Encoder untuk kolom binary
    'feature_columns'             : # Daftar 46 fitur setelah encoding
    'numeric_cols'                : # Kolom numerik yang di-scale
    'ohe_and_binary_encoded_cols' : # Kolom hasil OHE & binary encoding
}
```

> **Penting:** `model_artifacts.pkl` dihasilkan dari notebook modeling.
> Pastikan file ini ada sebelum menjalankan inference script.

---

#### 4.3.2.3  Cell 1 — Install & Import Library

```python
# Jalankan jika belum install
!pip install -q pandas scikit-learn

import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("✅ Library siap!")
```

---

#### 4.3.2.4  Cell 2 — Muat Model & Preprocessing Artifacts

```python
# ============================================================
# Muat model dan semua preprocessing artifacts
# Pastikan 'model_artifacts.pkl' ada di direktori yang sama
# ============================================================

import pickle
import pandas as pd

# Muat model dan preprocessing artifacts yang telah disimpan
# Pastikan path ke 'model_artifacts.pkl' disesuaikan jika file tidak berada di direktori yang sama
with open('model_artifacts.pkl', 'rb') as f:
    model_artifacts = pickle.load(f)

best_model      = model_artifacts['best_model']
best_model_name = model_artifacts['best_model_name']
scaler          = model_artifacts['scaler']
le_dict         = model_artifacts['le_dict']
feature_columns = model_artifacts['feature_columns']
numeric_cols    = model_artifacts['numeric_cols']
ohe_and_binary_encoded_cols = model_artifacts['ohe_and_binary_encoded_cols'] # Corrected variable name

print(f"✅ Model terbaik '{best_model_name}' dan preprocessing artifacts berhasil dimuat!")
```
---

#### 4.3.2.5 Cell 3 — Load Data Karyawan Baru

```python
# ============================================================
# Struktur kolom harus sama dengan data training
# (tanpa kolom Attrition)
# ============================================================

try:
    df_new_external = pd.read_csv('new_employee_data.csv')
    print(f"✅ Data baru dari 'new_employee_data.csv' berhasil dimuat. Shape: {df_new_external.shape}")
except FileNotFoundError:
    print("❌ File 'new_employee_data.csv' tidak ditemukan. Pastikan Anda mengunggahnya atau menyediakan path yang benar.")
    print("Menggunakan `df_unlabeled_backup` sebagai data baru untuk demo ini.")
    df_new_external = df_unlabeled_backup.copy()

df_new_unseen = df_new_external.copy()

print(f"✅ Data baru (df_new_unseen) siap untuk diproses lebih lanjut. Shape: {df_new_unseen.shape}")
print("Preview 5 baris pertama data baru:")
```

```

### Struktur kolom yang diperlukan

Data baru harus memiliki kolom berikut (tanpa kolom `Attrition`):

```python
kolom_wajib = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department',
    'DistanceFromHome', 'Education', 'EducationField',
    'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]
print(f"Kolom yang diperlukan: {len(kolom_wajib)} kolom")
```

---

#### 4.3.2.6 Cell 4 — Preprocessing Data Baru

```python
# ============================================================
# Terapkan preprocessing yang SAMA PERSIS dengan training:
# 1. Label Encoding untuk kolom binary (Gender, OverTime)
# 2. One-Hot Encoding untuk kolom nominal
# 3. Align kolom dengan feature_columns training
# 4. StandardScaler untuk kolom numerik
# ============================================================

# Identifikasi kolom kategorikal dan numerik di data baru
# Gunakan le_dict yang sudah dimuat untuk label encoding
for col, le in le_dict.items():
    if col in df_new_unseen.columns:
        df_new_unseen[col] = le.transform(df_new_unseen[col])

# Lakukan One-Hot Encoding untuk kolom nominal
# Dapatkan kembali daftar kolom nominal dari artifacts untuk memastikan konsistensi
nominal_cols = [c for c in ohe_and_binary_encoded_cols if not (c in le_dict or c in ['Gender', 'OverTime'])]

# Rekonstruksi kolom nominal asli dari daftar yang telah ditentukan jika diperlukan
# Bagian ini perlu kuat jika df_new_unseen memiliki kolom yang berbeda

# Identifikasi kolom kategorikal asli yang telah di-one-hot encode
original_nominal_features = []
for col_name in feature_columns:
    for prefix in ['BusinessTravel_', 'Department_', 'EducationField_', 'JobRole_', 'MaritalStatus_']:
        if col_name.startswith(prefix):
            original_nominal_features.append(prefix[:-1]) # Tambahkan nama fitur dasar
            break

original_nominal_features = list(set(original_nominal_features)) # Dapatkan nama dasar yang unik

# Pastikan original_nominal_features ada di df_new_unseen sebelum get_dummies
for col in original_nominal_features:
    if col not in df_new_unseen.columns:
        # Tangani kasus di mana kolom mungkin hilang di data baru (misalnya, jika semuanya nol)
        # Untuk saat ini, kita mengasumsikan semua kolom asli ada.
        pass

df_new_unseen_encoded = pd.get_dummies(df_new_unseen, columns=original_nominal_features, drop_first=False)

# Selaraskan kolom agar sama persis dengan X (data pelatihan) menggunakan feature_columns yang disimpan
X_new_unseen_processed = df_new_unseen_encoded.reindex(columns=feature_columns, fill_value=0)

# Ubah kolom boolean yang dibuat oleh get_dummies menjadi integer
for col in X_new_unseen_processed.select_dtypes(include='bool').columns:
    X_new_unseen_processed[col] = X_new_unseen_processed[col].astype(int)

# Terapkan scaling pada fitur numerik
X_new_unseen_processed[numeric_cols] = scaler.transform(X_new_unseen_processed[numeric_cols])

print(f"✅ Data baru berhasil dipreprocessing. Shape: {X_new_unseen_processed.shape}")
print("Preview data baru setelah preprocessing:")
display(X_new_unseen_processed.head())
```
---

#### 4.3.2.7 Cell 5 — Prediksi Attrition & Tampilkan Hasil

```python
# ============================================================
# Lakukan prediksi menggunakan model yang sudah dimuat
# ============================================================

y_new_pred = best_model.predict(X_new_unseen_processed)
y_new_prob = best_model.predict_proba(X_new_unseen_processed)[:, 1]

df_new_unseen['Attrition_Predicted'] = y_new_pred
df_new_unseen['Resign_Probability']  = y_new_prob.round(4)
df_new_unseen['Risk_Level'] = pd.cut(
    y_new_prob,
    bins   = [0, 0.3, 0.6, 1.0],
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
)

print("✅ Prediksi berhasil dilakukan pada data baru.")
print("Preview hasil prediksi (top 10 karyawan dengan probabilitas resign tertinggi):")

high_risk_new = (df_new_unseen[df_new_unseen['Risk_Level'] == 'High Risk']
                 .sort_values('Resign_Probability', ascending=False))

display(high_risk_new[['Resign_Probability', 'Risk_Level', 'Age',
                       'MonthlyIncome', 'OverTime', 'JobRole', 'MaritalStatus']].head(10))

print(f"Total karyawan yang diprediksi 'High Risk': {len(high_risk_new)} orang")
```
---

#### 4.3.2.7 Cell 6 — Export Hasil

```python
# ============================================================
# Tampilkan karyawan berisiko tinggi & simpan hasil
# ============================================================

# Ekspor hasil prediksi ke file CSV
# Anda bisa mengganti 'prediction_results.csv' dengan nama file yang diinginkan
output_filename = 'prediction_results.csv'
df_new_unseen.to_csv(output_filename, index=False)

print(f"✅ Hasil prediksi berhasil diekspor ke '{output_filename}'")
print("File dapat diunduh dari panel File di sebelah kiri Colab.")
```

---

#### 4.3.2.8 Catatan Penting

> Model yang digunakan adalah **XGBoost** dengan performa:
> - Recall : **0.52** (75% karyawan resign berhasil terdeteksi)
> - ROC-AUC: **0.81**
>
> Model ini dilatih menggunakan data PT Jaya Jaya Maju.
> Untuk data perusahaan lain atau periode berbeda, model perlu **dilatih ulang**.

---

## 5. Business Dashboard

### 5.1 Akses Dashboard

| Akses | Detail |
|---|---|
| URL | `http://localhost:3000` |
| Email | `root@mail.com` |
| Password | `root123` |

---

### 5.2 Gambaran Umum Dashboard

Dashboard ini dirancang untuk membantu **Manajer HR PT Jaya Jaya Maju** dalam memonitor dan memahami faktor-faktor yang mempengaruhi tingginya attrition rate perusahaan. Dashboard terdiri dari **13 visualisasi** yang dibagi ke dalam 5 bagian utama, mencakup data **1.470 karyawan** — gabungan dari 1.058 data berlabel aktual dan 412 data hasil prediksi model Machine Learning (XGBoost, Recall=0.52, ROC-AUC=0.81).

---

### 5.3 Fitur Filter Interaktif

Di bagian paling atas dashboard terdapat **4 filter interaktif** yang memungkinkan HR melihat data dari berbagai sudut pandang:

| Filter | Variabel | Nilai yang Bisa Diketik |
|---|---|---|
| Departemen | `filter_department` | `Human Resources` · `Research & Development` · `Sales` |
| Risk Level | `filter_risklevel` | `High Risk` · `Medium Risk` · `Low Risk` |
| Overtime | `filter_overtime` | `Yes` · `No` |
| Kelompok Usia | `filter_agegroup` | `18-25` · `26-35` · `36-45` · `46-60` |

Saat salah satu filter diisi, **seluruh visualisasi di dashboard akan otomatis menyesuaikan** dan hanya menampilkan data sesuai segmen yang dipilih. Kosongkan filter untuk kembali ke tampilan keseluruhan 1.470 karyawan.

---

### 5.4 Bagian 1 — KPI Summary (4 Kartu Angka)

Baris pertama berisi 4 KPI utama yang memberikan gambaran cepat kondisi attrition perusahaan.

#### 5.4.1 Total Karyawan: **1.470**
Jumlah total seluruh karyawan dalam dataset, mencakup 1.058 data berlabel aktual dan 412 data yang diprediksi oleh model ML.

#### 5.4.2 Attrition Rate Aktual (%): **16.9**
Persentase karyawan yang benar-benar telah resign berdasarkan data berlabel. Angka ini jauh melampaui target maksimal perusahaan sebesar 10%, menunjukkan adanya masalah retensi yang perlu segera ditangani.

#### 5.4.3 Total Prediksi Resign (Aktual + Model ML): **235**
Gabungan karyawan yang sudah resign (data aktual) dan karyawan yang diprediksi akan resign oleh model ML. Angka ini mencerminkan **total risiko resign** yang dihadapi perusahaan saat ini.

#### 5.4.4 Karyawan High Risk: **248**
Jumlah karyawan yang masuk kategori High Risk berdasarkan RiskScore — kombinasi faktor overtime, gaji rendah, job involvement rendah, environment satisfaction rendah, work-life balance buruk, tidak punya stock option, masa kerja singkat, dan sering perjalanan dinas. Karyawan ini membutuhkan **intervensi HR segera**.

---

### 5.5 Bagian 2 — Distribusi Populasi (2 Pie Chart)

#### 5.5.1 Proporsi Stay vs Resign — 1.470 Karyawan
Pie chart ini menampilkan proporsi karyawan yang Stay (84%) vs Resign (16%) dari keseluruhan 1.470 karyawan, menggunakan kolom `Attrition_Combined_Label` yang menggabungkan data aktual dan prediksi model ML. Ketidakseimbangan kelas yang signifikan (rasio 5:1) inilah yang menjadi alasan penggunaan SMOTE saat training model.

#### 5.5.2 Distribusi Risk Level — 1.470 Karyawan
Pie chart ini menampilkan distribusi tingkat risiko seluruh karyawan:
- **Low Risk: 44.2%** (649 karyawan) — aman, monitoring rutin bulanan
- **Medium Risk: 38.9%** (572 karyawan) — perlu monitoring aktif 2 minggu sekali
- **High Risk: 16.9%** (248 karyawan) — perlu tindakan segera dalam 5 hari kerja

---

### 5.6 Bagian 3 — Analisis per Job Role & Departemen

#### 5.6.1 Attrition Rate per Job Role (%)
Bar chart horizontal ini menampilkan attrition rate untuk setiap posisi jabatan, diurutkan dari tertinggi ke terendah:

| Job Role | Attrition Rate | Interpretasi |
|---|---|---|
| Sales Representative | **43.1%** | Kritis — hampir separuh resign |
| Laboratory Technician | **26.1%** | Sangat tinggi |
| Human Resources | **20.0%** | Tinggi |
| Research Scientist | **17.8%** | Di atas rata-rata |
| Sales Executive | **16.8%** | Di atas rata-rata |
| Healthcare Representative | **9.1%** | Mendekati target |
| Manufacturing Director | **6.5%** | Baik |
| Manager | **6.3%** | Baik |
| Research Director | **3.2%** | Sangat baik |

Sales Representative menjadi posisi paling kritis dengan attrition rate 43.1% — hampir 3x rata-rata perusahaan. Ini mengindikasikan perlunya evaluasi khusus pada kondisi kerja, kompensasi, dan beban kerja di posisi ini.

#### 5.6.2 Attrition Rate per Departemen (%)
Bar chart ini menampilkan perbandingan attrition rate antar departemen:
- **Sales: 20.7%** — tertinggi, jauh di atas rata-rata
- **Human Resources: 15.8%** — di atas rata-rata
- **Research & Development: 15.3%** — sedikit di bawah rata-rata

Departemen Sales menjadi prioritas utama intervensi karena kombinasi attrition rate tertinggi dan jumlah karyawan yang signifikan (446 orang).

---

### 5.7 Bagian 4 — Analisis Faktor Risiko

#### 5.7.1 Attrition Rate per Kelompok Gaji (%)
Bar chart ini membuktikan korelasi negatif yang kuat antara gaji dan attrition:
- **< 3K: 28.7%** — hampir 3x rata-rata perusahaan
- **3K–6K: 15.1%** — sedikit di bawah rata-rata
- **6K–10K: 11.7%** — mendekati target
- **> 10K: 9.1%** — sudah di bawah target 10%

Karyawan bergaji rendah (< 3K) memiliki risiko resign paling tinggi. Ini menjadi dasar rekomendasi salary benchmarking dan penyesuaian struktur gaji minimum.

#### 5.7.2 Attrition Rate per Kelompok Usia (%)
Bar chart ini menampilkan pola attrition berdasarkan kelompok usia:
- **18–25 tahun: 38%** — sangat kritis, fase adaptasi awal karier
- **26–35 tahun: 19.5%** — masih di atas rata-rata
- **36–45 tahun: 11%** — mendekati target
- **46–60 tahun: 11.6%** — mendekati target

Karyawan muda (18–25 tahun) memiliki risiko resign tertinggi, mengindikasikan perlunya program onboarding yang lebih kuat dan jalur karier yang jelas untuk generasi awal.

#### 5.7.3 Dampak Overtime terhadap Attrition (%)
Bar chart ini menampilkan salah satu temuan paling signifikan:
- **Overtime Yes: 31.9%** — hampir 3x lipat
- **Overtime No: 10.8%** — sudah di bawah target perusahaan

Selisih 21.1 percentage points ini menjadikan overtime sebagai **faktor risiko tunggal terbesar** yang mempengaruhi attrition. Dari 307 karyawan yang overtime, 98 di antaranya resign.

---

### 5.8 Bagian 5 — Analisis Prediktif

#### 5.8.1 Distribusi Probabilitas Resign Karyawan
Bar chart ini menampilkan distribusi probabilitas resign hasil model ML untuk seluruh 1.470 karyawan:
- **Sangat Rendah (< 0.2): 1.172 karyawan** — mayoritas relatif aman
- **Rendah (0.2–0.4): 46 karyawan** — monitoring ringan
- **Sedang (0.4–0.6): 27 karyawan** — perlu perhatian
- **Tinggi (0.6–0.8): 18 karyawan** — risiko serius
- **Sangat Tinggi (≥ 0.8): 207 karyawan** — intervensi mendesak

207 karyawan dengan probabilitas resign ≥ 80% menjadi prioritas utama tindakan HR dalam jangka pendek.

#### 5.8.2 Top 50 Karyawan Berisiko Tinggi (Tabel)
Tabel ini menampilkan 50 karyawan dengan RiskScore dan Resign_Probability tertinggi, dilengkapi informasi:
- Data demografis: usia, departemen, jabatan
- Kondisi kerja: gaji bulanan, overtime, masa kerja
- Tingkat kepuasan: job satisfaction, environment satisfaction
- Status prediksi: status aktual (jika berlabel) dan status prediksi model ML

Dari tabel terlihat pola konsisten — karyawan berisiko tinggi umumnya berasal dari R&D (Laboratory Technician) dan Sales (Sales Representative), bergaji di bawah 3.000, melakukan overtime, dan memiliki job satisfaction serta environment satisfaction yang rendah.

---

### 5.9 Panduan Filter

```
Departemen : Human Resources | Research & Development | Sales
Risk Level : High Risk | Medium Risk | Low Risk
Overtime   : Yes | No
Kel. Usia  : 18-25 | 26-35 | 36-45 | 46-60
(Kosongkan untuk tampilkan semua data)
```

---

### 5.10 Kesimpulan Dashboard

Dashboard ini berhasil menjawab 5 pertanyaan bisnis utama yang ditetapkan di awal proyek:

| Pertanyaan Bisnis | Jawaban dari Dashboard |
|---|---|
| Faktor apa yang paling mempengaruhi attrition? | Overtime (31.9%), usia muda (38%), gaji rendah (28.7%) |
| Ada pola per departemen/role? | Sales & Lab Technician paling kritis |
| Profil karyawan berisiko tinggi? | Terlihat di tabel Top 50 High Risk |
| Seberapa akurat prediksi model? | Recall 0.52, ROC-AUC 0.83 |
| Intervensi HR yang efektif? | Kontrol overtime, naikkan gaji < 3K, program retensi muda |

---

## 6. Conclusion

Berdasarkan analisis data 1.470 karyawan PT Jaya Jaya Maju, attrition rate perusahaan tercatat sebesar 16,9% — jauh melampaui target maksimal 10%. Faktor-faktor utama yang mempengaruhi tingginya attrition rate adalah overtime (karyawan yang lembur memiliki attrition 31,9% vs 10,8% yang tidak lembur), job role (Sales Representative 43,1%), usia muda 18–25 tahun (37,2%), masa kerja kurang dari 1 tahun (34,6%), serta gaji rendah di bawah 3.000 (28,7%). Model Machine Learning XGBoost berhasil dibangun dengan performa Recall 0,52 dan ROC-AUC 0,81 sehingga model mampu mendeteksi 52% karyawan yang berisiko resign sebelum keputusan tersebut diambil. Untuk mengatasi permasalahan ini, perusahaan direkomendasikan untuk segera mengendalikan kebijakan overtime, memperkuat program retensi karyawan muda dan baru, melakukan salary benchmarking, meningkatkan job involvement, serta memperjelas jalur karier — dengan target penurunan attrition rate dari 16,9% menjadi di bawah 10% dalam 12 bulan.

---

## 7. Rekomendasi Action Items

Berdasarkan hasil analisis model prediksi *attrition* dan identifikasi faktor-faktor paling berpengaruh, berikut adalah rekomendasi *action items* yang dapat dilakukan oleh departemen HR Jaya Jaya Maju untuk menurunkan *attrition rate*:

---

#### 1. Targeted Retention Program (Prioritas Utama)
*   **Fokus:** Prioritaskan intervensi pada **248 karyawan yang teridentifikasi `High Risk`**. Lakukan pertemuan *one-on-one* untuk memahami kekhawatiran dan kebutuhan mereka secara personal.
*   **Aksi:** Tawarkan solusi personal seperti jalur pengembangan karir yang lebih jelas, mentoring, penyesuaian beban kerja, atau evaluasi kompensasi.

#### 2. Pengembangan Karir & Mentoring
*   **Fokus:** Karyawan muda (terutama di bawah 30 tahun) dan karyawan pada *job role* berisiko tinggi (misalnya `Sales Representative`, `Laboratory Technician`).
*   **Aksi:** Sediakan program *mentorship*, pelatihan keterampilan (*upskilling/reskilling*), dan kesempatan pengembangan karir yang transparan untuk meningkatkan keterlibatan dan prospek masa depan mereka di perusahaan.

#### 3. Peningkatan Kompensasi & Benefit
*   **Fokus:** Karyawan dengan `MonthlyIncome` yang relatif rendah, terutama jika mereka teridentifikasi berisiko tinggi.
*   **Aksi:** Lakukan studi *salary benchmark* secara berkala untuk memastikan kompensasi kompetitif. Pertimbangkan penyesuaian gaji atau penawaran benefit lain yang menarik untuk *job role* tertentu.
*   **Insight:** Tingkat `StockOptionLevel` yang rendah juga menjadi faktor. Evaluasi program kepemilikan saham karyawan jika ada, atau cara lain untuk meningkatkan rasa kepemilikan.

#### 4. Manajemen Beban Kerja & Keseimbangan Hidup-Kerja (Work-Life Balance)
*   **Fokus:** Karyawan yang sering lembur (`OverTime = Yes`) dan mereka yang menunjukkan `WorkLifeBalance` atau `JobInvolvement` rendah.
*   **Aksi:** Terapkan kebijakan yang lebih ketat atau memberikan kompensasi yang adil untuk lembur. Adakan program *well-being* karyawan, fleksibilitas jam kerja, atau opsi kerja *hybrid* untuk mengurangi *burnout* dan meningkatkan keseimbangan hidup-kerja.

---
