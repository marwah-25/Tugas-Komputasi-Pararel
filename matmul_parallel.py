"""
Perkalian matriks C = A x B : versi serial dan paralel (data parallelism).
Paralel: multiprocessing.Pool + shared_memory, baris C dibagi ke P proses.
Serial dan paralel memakai kernel hitung yang sama agar perbandingan adil.
"""
import numpy as np
from multiprocessing import Pool, shared_memory


def kernel_baris(baris_a, bt):
    """Hitung sekelompok baris C. baris_a: list baris A, bt: B transpose (list)."""
    return [[sum(x * y for x, y in zip(ra, kb)) for kb in bt] for ra in baris_a]


def matmul_serial(A, B):
    return np.array(kernel_baris(A.tolist(), B.T.tolist()))


def _worker(args):
    nama_a, nama_b, nama_c, n, r0, r1 = args
    sa = shared_memory.SharedMemory(name=nama_a)
    sb = shared_memory.SharedMemory(name=nama_b)
    sc = shared_memory.SharedMemory(name=nama_c)
    A = np.ndarray((n, n), dtype=np.float64, buffer=sa.buf)
    B = np.ndarray((n, n), dtype=np.float64, buffer=sb.buf)
    C = np.ndarray((n, n), dtype=np.float64, buffer=sc.buf)
    C[r0:r1, :] = kernel_baris(A[r0:r1].tolist(), B.T.tolist())  # tiap proses menulis barisnya sendiri
    for s in (sa, sb, sc):
        s.close()


def matmul_parallel(A, B, p):
    n = A.shape[0]
    shm = [shared_memory.SharedMemory(create=True, size=A.nbytes) for _ in range(3)]
    try:
        for s, M in zip(shm[:2], (A, B)):
            np.ndarray(M.shape, dtype=np.float64, buffer=s.buf)[:] = M
        batas = np.linspace(0, n, p + 1, dtype=int)  # pembagian baris
        tugas = [(shm[0].name, shm[1].name, shm[2].name, n, int(batas[i]), int(batas[i + 1]))
                 for i in range(p)]
        with Pool(p) as pool:
            pool.map(_worker, tugas)
        return np.ndarray(A.shape, dtype=np.float64, buffer=shm[2].buf).copy()
    finally:
        for s in shm:
            s.close()
            s.unlink()


if __name__ == "__main__":
    import time, sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(42)
    A, B = rng.random((n, n)), rng.random((n, n))
    acuan = A @ B
    t = time.perf_counter(); Cs = matmul_serial(A, B); ts = time.perf_counter() - t
    print(f"Serial     : {ts:.4f} s | benar = {np.allclose(Cs, acuan)}")
    for p in (2, 4, 8):
        t = time.perf_counter(); Cp = matmul_parallel(A, B, p); tp = time.perf_counter() - t
        print(f"Paralel p={p}: {tp:.4f} s | speedup {ts/tp:.3f}x | efisiensi {ts/tp/p*100:.1f}% | benar = {np.allclose(Cp, acuan)}")
