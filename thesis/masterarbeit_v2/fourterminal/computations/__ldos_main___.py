import scharf_ldos as S
import numpy as np

Ez_val = 0.4                     
B_val  = S.PP.B_of_E_Z(Ez_val)   
S.PP.summary()
print(f"E_Z = {Ez_val} meV  ->  B = {B_val:.2f} T   ({Ez_val/S.PP.E_T:.2f} E_T)")

R = S.run_map_parallel(n_workers=8, save="scharf_map.npz")
S.plot_map(R)
S.plot_curvature(R)

x, y, A = S.real_space(np.pi, B_val)

E, phis, end, mid = S.spectrum(
    B_val,
    E_meV=np.linspace(-0.3, 0.3, 61),
    phis=np.linspace(0, 2*np.pi, 41),
)
