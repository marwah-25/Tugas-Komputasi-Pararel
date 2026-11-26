"""
Benchmark perkalian matriks serial vs paralel.
Tiap pengukuran: 1x pemanasan (dibuang), 3x ulang, ambil median.
Eksperimen A : N tetap, jumlah proses 2/4/8.   Eksperimen B : proses tetap, N bervariasi.
Keluaran: hasil_benchmark.csv, info_sistem.json, grafik_hasil.png
Pemakaian: py benchmark.py [N_eksperimen_A]   (default 300)
"""
import csv, json, platform, statistics, sys, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matmul_parallel import matmul_serial, matmul_parallel


def ukur(fungsi, ulang=3):
    fungsi()  # pemanasan
    waktu = []
    for _ in range(ulang):
        t = time.perf_counter(); fungsi(); waktu.append(time.perf_counter() - t)
    return statistics.median(waktu)


def data(n):
    rng = np.random.default_rng(42)
    return rng.random((n, n)), rng.random((n, n))


if __name__ == "__main__":
    N_A = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    P_B = 8
    ukuran_B = [100, 200, N_A]
    baris = []
    print(f"CPU logical: {os.cpu_count()}")

    # Eksperimen A
    A, B = data(N_A); acuan = A @ B
    ts = ukur(lambda: matmul_serial(A, B))
    baris.append(["A", N_A, 1, ts, 1.0, 100.0, 0.0, True])
    print(f"[A] serial {ts:.4f}s")
    for p in (2, 4, 8):
        tp = ukur(lambda: matmul_parallel(A, B, p))
        ok = bool(np.allclose(matmul_parallel(A, B, p), acuan))
        sp = ts / tp
        baris.append(["A", N_A, p, tp, sp, sp / p * 100, (ts - tp) / ts * 100, ok])
        print(f"[A] p={p} {tp:.4f}s speedup {sp:.3f}x efisiensi {sp/p*100:.1f}% benar={ok}")

    # Eksperimen B
    for n in ukuran_B:
        A, B = data(n); acuan = A @ B
        ts = ukur(lambda: matmul_serial(A, B))
        tp = ukur(lambda: matmul_parallel(A, B, P_B))
        ok = bool(np.allclose(matmul_parallel(A, B, P_B), acuan))
        sp = ts / tp
        baris.append(["B", n, P_B, tp, sp, sp / P_B * 100, (ts - tp) / ts * 100, ok, ts])
        print(f"[B] N={n} serial {ts:.4f}s paralel {tp:.4f}s speedup {sp:.3f}x benar={ok}")

    with open("hasil_benchmark.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eksperimen", "N", "proses", "waktu_detik", "speedup", "efisiensi_persen", "pengurangan_waktu_persen", "hasil_sama", "waktu_serial_B"])
        w.writerows(baris)
    with open("info_sistem.json", "w") as f:
        json.dump({"os": platform.platform(), "python": platform.python_version(), "cpu": platform.processor(),
                   "cpu_logical": os.cpu_count(), "numpy": np.__version__}, f, indent=2)

    # Grafik
    A_ = [b for b in baris if b[0] == "A"]; B_ = [b for b in baris if b[0] == "B"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar([str(b[2]) for b in A_], [b[3] for b in A_], color="#1d4ed8")
    ax[0].set(title="Waktu eksekusi", xlabel="Jumlah proses (1 = serial)", ylabel="Detik")
    ax[1].plot([b[2] for b in A_], [b[4] for b in A_], "o-", label="Hasil pengujian")
    ax[1].plot([1, 8], [1, 8], "--", color="gray", label="Ideal")
    ax[1].set(title="Speedup", xlabel="Jumlah proses", ylabel="Speedup"); ax[1].legend()
    ax[2].plot([b[1] for b in B_], [b[4] for b in B_], "o-", color="#0fa3b1")
    ax[2].axhline(1, ls=":", color="gray")
    ax[2].set(title=f"Speedup vs ukuran matriks (p={P_B})", xlabel="Ukuran N", ylabel="Speedup")
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("grafik_hasil.png", dpi=150)
    print("Selesai: hasil_benchmark.csv, info_sistem.json, grafik_hasil.png")
