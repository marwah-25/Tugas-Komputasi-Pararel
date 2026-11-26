# Perkalian Matriks Paralel (Tugas Komputasi Paralel dan Terdistribusi)

**Nama:** Marwah | **NIM:** D0224303 | **Kelas:** [ISI KELAS]

Perbandingan perkalian matriks C = A x B secara **serial** dan **paralel** menggunakan Python
(`multiprocessing.Pool` + `shared_memory`).

## Isi repositori
| File | Keterangan |
|---|---|
| `matmul_parallel.py` | Fungsi perkalian matriks serial dan paralel + uji cepat |
| `benchmark.py` | Pengukuran waktu (pemanasan 1x, 3x ulang, median), menghasilkan CSV dan grafik |
| `Colab_Matriks_Paralel.ipynb` | Versi notebook Google Colab (semua kode dalam satu file) |
| `hasil_benchmark.csv`, `grafik_hasil.png` | Hasil pengujian (tambahkan dari Colab) |

## Cara kerja paralel
1. Matriks A, B, dan C ditempatkan di shared memory (tanpa menyalin ke tiap proses).
2. Baris matriks C dibagi rata ke P proses (`np.linspace(0, n, P+1)`).
3. Tiap proses menghitung blok barisnya sendiri dan menulis ke baris yang berbeda (tanpa lock).
4. Hasil disalin dan diverifikasi dengan `A @ B` (NumPy).

Speedup = T_serial / T_paralel, Efisiensi = Speedup / P x 100%.

## Cara menjalankan
**Google Colab:** upload `Colab_Matriks_Paralel.ipynb`, lalu Runtime > Run all.

**Komputer lokal:**
```bash
pip install numpy matplotlib
python matmul_parallel.py 200     # uji cepat serial vs paralel
python benchmark.py 300           # benchmark lengkap
```

## Hasil (Google Colab, matriks 300x300)
| Proses | Waktu (s) | Speedup | Efisiensi |
|---|---|---|---|
| Serial | 3,0396 | 1,000x | 100,0% |
| 2 | 3,1175 | 0,975x | 48,8% |
| 4 | 2,8078 | 1,083x | 27,1% |
| 8 | 2,5876 | 1,175x | 14,7% |

Hasil paralel identik dengan hasil NumPy pada seluruh pengujian. Speedup dibatasi oleh overhead
dan jumlah core pada mesin Colab.

## Catatan penggunaan AI
Kerangka kode dan struktur laporan dibuat dengan bantuan AI (Claude, Anthropic).
Program dijalankan, diukur, dan hasilnya dianalisis sendiri.
