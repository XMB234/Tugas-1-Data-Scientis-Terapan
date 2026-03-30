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

### 3.1 Deliverable Proyek

| No. | Komponen | Deskripsi | Keterangan |
|---|---|---|---|
| 1 | Data Understanding | Eksplorasi awal dataset: shape, tipe data, distribusi, missing values | Selesai di awal proyek |
| 2 | Data Preparation | Handling missing values, encoding, feature selection, train-test split | Google Colab |
| 3 | Modeling | Membangun model prediksi attrition (klasifikasi biner: 0 = stay, 1 = resign) | Google Colab |
| 5 | Evaluation | Mengukur performa model: Accuracy, Precision, Recall, F1-Score, ROC-AUC | Google Colab |
| 6 | Business Dashboard | Dashboard interaktif untuk monitoring faktor attrition secara real-time | Metabase / Tableau |
| 7 | Rekomendasi Bisnis | Actionable insight berbasis hasil model dan EDA untuk tim HR | Laporan akhir |

### 3.2 Batasan Proyek (*Out of Scope*)

- Implementasi perubahan kebijakan HR secara langsung — proyek ini hanya memberikan rekomendasi berbasis data.
- Pengumpulan data primer tambahan di luar dataset yang telah disediakan perusahaan.
- Pengembangan sistem HR berbasis produksi (*production-ready system*) — fokus pada analisis dan prototipe.
- Analisis finansial mendalam terkait biaya attrition — hanya estimasi kualitatif.

### 3.3 Asumsi Proyek

- Dataset yang disediakan merepresentasikan kondisi aktual karyawan perusahaan secara akurat.
- Kolom `Attrition` bernilai `1` = karyawan telah resign, `0` = karyawan masih aktif; baris dengan nilai kosong diasumsikan sebagai data yang belum dilabeli.
- Kolom `EmployeeCount`, `Over18`, dan `StandardHours` dikonfirmasi sebagai fitur konstan dan akan di-drop pada tahap preprocessing.
- Model yang dibangun ditujukan untuk mendukung keputusan HR, bukan sebagai pengganti pertimbangan manusiawi.

### 3.4 Target Keberhasilan Proyek

| Komponen | Target |
|---|---|
| Model ML | ROC-AUC ≥ 0,80 |
| Business Dashboard | Minimal 6 dimensi analisis attrition secara interaktif |
| Rekomendasi Bisnis | Minimal 5 rekomendasi konkret dan terukur |

---

