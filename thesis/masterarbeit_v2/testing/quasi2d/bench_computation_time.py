#%%
import os
import time
import numpy as np
import scipy.linalg as la
from joblib import Parallel, delayed

# ==========================================
# 1. SETUP PARAMETERS & SYSTEM SIZES
# ==========================================
ny = 20
normal = 5
nbar = 1
it = 1500
t = 1.0
Delta = 0.25 / 11.2  # Scaled dummy values matching parameters
eta = 0.01 * Delta
nphi = 71

phi_vals = np.linspace(0, 2 * np.pi, nphi)
_I4 = np.eye(4 * ny, dtype=np.complex128)
_elec_idx = np.array([i for i in range(4 * ny) if i % 4 <  2])
_hole_idx = np.array([i for i in range(4 * ny) if i % 4 >= 2])

# Use a smaller slice of energies for a quick benchmark
benchmark_energies = np.linspace(-1.01 * Delta, 1.01 * Delta, 10)

# ==========================================
# 2. CORE HELPER FUNCTIONS (DENSE & VECTORIZED)
# ==========================================
def dummy_hamiltonians():
    """Generates dummy dense matrices of size 400x400 to mimic real system."""
    dim = 4 * ny
    Hs = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Hs = (Hs + Hs.conj().T) * 0.1  # Hermitizing
    Hb = Hs.copy()
    Hn = Hs.copy()
    V0 = np.random.randn(dim, dim) * 0.05 + 1j * np.random.randn(dim, dim) * 0.05
    Vd = V0.conj().T
    return Hs, Hb, Hn, V0, Vd

def sancho(H, alpha_0, beta_0, w, eta, Id):
    z = w + 1j * eta
    zI = z * Id
    eps, eps_s, alpha, beta = H.copy(), H.copy(), alpha_0.copy(), beta_0.copy()
    tol = 1e-10
    for _ in range(200): # Hard cap for quick diagnostic benchmark
        g = np.linalg.solve(zI - eps, Id)
        ab, ba = alpha @ g @ beta, beta @ g @ alpha
        eps += ab + ba
        eps_s += ab
        alpha, beta = alpha @ g @ alpha, beta @ g @ beta
        if max(np.max(np.abs(alpha)), np.max(np.abs(beta))) < tol:
            break
    return np.linalg.solve(zI - eps_s, Id)

def _apply_phase_vectorized(gSR, phi_array):
    nphi = len(phi_array)
    G_out = np.tile(gSR, (nphi, 1, 1))
    ep = np.exp( 1j * phi_array)[:, None, None]
    em = np.exp(-1j * phi_array)[:, None, None]
    G_out[:, _elec_idx[:, None], _hole_idx] *= ep
    G_out[:, _hole_idx[:, None], _elec_idx] *= em
    return G_out

def _ldos_one_slice_vectorized(w, Hs, Hb, Hn, V0, Vd):
    z = w + 1j * eta
    dim = Hs.shape[0]
    Id = np.eye(dim, dtype=np.complex128)
    zI = z * Id

    gSL0 = sancho(Hs, V0, Vd, w, eta, Id)
    gSR0 = sancho(Hs, Vd, V0, w, eta, Id)

    gSL = np.linalg.solve(zI - Hb - V0 @ gSL0 @ Vd, Id)
    gSR = np.linalg.solve(zI - Hb - Vd @ gSR0 @ V0, Id)

    glr = np.empty((normal, dim, dim), dtype=np.complex128)
    glr[0] = np.linalg.solve(zI - Hn - V0 @ gSL @ Vd, Id)
    for i in range(1, normal):
        glr[i] = np.linalg.solve(zI - Hn - V0 @ glr[i-1] @ Vd, Id)

    SL = np.empty_like(glr)
    SL[0] = V0 @ gSL @ Vd
    for i in range(1, normal):
        SL[i] = V0 @ glr[i-1] @ Vd

    gSRp = _apply_phase_vectorized(gSR, phi_vals)
    grl = np.empty((normal, nphi, dim, dim), dtype=np.complex128)
    grl[-1] = np.linalg.solve(zI - Hn - Vd @ gSRp @ V0, Id)
    for j in range(normal - 2, -1, -1):
        grl[j] = np.linalg.solve(zI - Hn - Vd @ grl[j+1] @ V0, Id)

    result = np.zeros(nphi)
    for j in range(normal):
        SR = Vd @ grl[j+1] @ V0 if j < normal - 1 else Vd @ gSRp @ V0
        G = np.linalg.solve(zI - Hn - SL[j] - SR, Id)
        result += -np.imag(np.trace(G, axis1=1, axis2=2)) / np.pi

    return result

# ==========================================
# 3. BENCHMARK DRIVERS
# ==========================================
def run_sequential(energy_array, Hs, Hb, Hn, V0, Vd):
    """Executes the loop purely sequentially on a single thread."""
    results = []
    for w in energy_array:
        results.append(_ldos_one_slice_vectorized(w, Hs, Hb, Hn, V0, Vd))
    return np.array(results)

def run_joblib(energy_array, Hs, Hb, Hn, V0, Vd):
    """Executes the loop using all available CPU cores via Joblib."""
    results = Parallel(n_jobs=-1)(
        delayed(_ldos_one_slice_vectorized)(w, Hs, Hb, Hn, V0, Vd)
        for w in energy_array
    )
    return np.array(results)


if __name__ == '__main__':
    print("=" * 60)
    print("        DIAGNOSTIC BENCHMARK: JOBLIB VS SEQUENTIAL")
    print("=" * 60)
    
    # 1. Warm-up and Hamiltonian Generation
    Hs, Hb, Hn, V0, Vd = dummy_hamiltonians()
    
    # 2. Timing Sequential Implementation
    # We enforce MKL single-threading inside the sequential wrapper to get a true comparison
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    print(f"Running sequential execution for {len(benchmark_energies)} energy steps...")
    t0 = time.time()
    res_seq = run_sequential(benchmark_energies, Hs, Hb, Hn, V0, Vd)
    t_seq = time.time() - t0
    print(f"⏱️  Sequential Time: {t_seq:.4f} seconds")
    print()
    
    # 3. Timing Joblib Parallelization
    print(f"Running joblib parallel execution (all available cores)...")
    t0 = time.time()
    res_job = run_joblib(benchmark_energies, Hs, Hb, Hn, V0, Vd)
    t_job = time.time() - t0
    print(f"⏱️  Joblib Parallel Time: {t_job:.4f} seconds")
    print()
    
    # ==========================================
    # 4. DIAGNOSTIC METRICS
    # ==========================================
    speedup = t_seq / t_job
    numerical_match = np.allclose(res_seq, res_job, rtol=1e-8, atol=1e-8)
    
    print("=" * 60)
    print("DIAGNOSTIC RESULTS Summary:")
    print("=" * 60)
    print(f"• Total Speedup Factor       : {speedup:.2f}x")
    print(f"• Array Data Identity Check  : {'PASS ✅' if numerical_match else 'FAIL ❌'}")
    
    if speedup > 1.2:
         print("\nConclusion: Joblib provides a clean scaling benefit across the energy sweep.")
    else:
         print("\nConclusion: Overhead dominated. For small energy counts, sequential may perform closely to joblib.")
    print("=" * 60)
# %%
