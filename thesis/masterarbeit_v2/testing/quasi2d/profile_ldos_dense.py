import importlib.util
import cProfile
import pstats
import io

# load dense module
spec = importlib.util.spec_from_file_location("quasi2_dense", r"C:\coding\my_projects\thesis\masterarbeit_v2\scripts\quasi_2d_benni.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# build Hamiltonians
alpha_raw = 14.3e-9
beta_raw = 7.3e-9
B_val = 2.3
Hs, Hb, Hn, V0, Vd = mod._build_hamiltonians(alpha_raw, beta_raw, B_field=B_val)

w = 0.0

pr = cProfile.Profile()
pr.enable()
res = mod._ldos_one_slice(w, Hs, Hb, Hn, V0, Vd, spatial=False)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(40)
print(s.getvalue())

# Also print a brief summary of times for major steps by re-running with manual timers
import time
print('\nManual timing breakdown:')
start = time.perf_counter()
import time
# Surface GF (sancho) timing
t0 = time.perf_counter()
gSL0 = mod.sancho(Hs, Vd, V0, w, mod.eta)
t1 = time.perf_counter()
gSR0 = mod.sancho(Hs, V0, Vd, w, mod.eta)
t2 = time.perf_counter()
print('sancho left', t1-t0, 'sancho right', t2-t1)

# build and solve gSL/gSR
z = w + 1j*mod.eta
zI = z * mod._I4
mat1 = zI - Hb - Vd @ gSL0 @ V0
mat2 = zI - Hb - V0 @ gSR0 @ Vd
t3 = time.perf_counter()
gSL = mod.la.solve(mat1, mod.np.eye(mat1.shape[0], dtype=complex))
gSR = mod.la.solve(mat2, mod.np.eye(mat2.shape[0], dtype=complex))
t4 = time.perf_counter()
print('gSL/gSR solve time', t4-t3)

# left-right chain timing
mat = zI - Hn - Vd @ gSL @ V0
t5 = time.perf_counter()
glr0 = mod.la.solve(mat, mod.np.eye(mat.shape[0], dtype=complex))
chain_time = 0.0
prev = glr0
for i in range(1, mod.normal):
    mat = zI - Hn - Vd @ prev @ V0
    t_s = time.perf_counter()
    prev = mod.la.solve(mat, mod.np.eye(mat.shape[0], dtype=complex))
    chain_time += time.perf_counter() - t_s
print('left-right chain solve initial', time.perf_counter()-t5, 'iterative chain time', chain_time)

print('Total manual elapsed', time.perf_counter()-start)
