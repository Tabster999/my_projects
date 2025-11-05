#%%
import numpy as np 
import matplotlib.pyplot as plt
from scipy import constants as const
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import trapezoid
#%%
gy_data = np.loadtxt(r"C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\SS25\Magnetismus\Dy.txt")

gy_xas = gy_data[:,0]
gy_xmcd = gy_data[:,1]
gy_wl = gy_data[:,2]
# %%
figsize = (10,7)
dpi = 150 

plt.figure(figsize=figsize, dpi=dpi)
plt.plot(gy_wl,gy_xas, label='XAS', c='blue')
plt.plot(gy_wl,gy_xmcd, label='XMCD', c='orange')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
#bc of order of exercise we firts ommit T_z but take it into account later (or discuss if it's negligable)
#%% integration to get the values of p and q
m5_range = (gy_wl > 1280) & (gy_wl < 1310)
m4_range = (gy_wl > 1310) & (gy_wl < 1340)

p = trapezoid(gy_xmcd[m5_range], gy_wl[m5_range])

#q: XMCD over M4+M5
q = trapezoid(gy_xmcd, gy_wl)

#r: XAS over M4+M5
r = trapezoid(gy_xas, gy_wl)
pp = trapezoid(gy_xas[m5_range], gy_wl[m5_range])

print(f"p={p};  q={q};   r={r}")

S_z = -((5*p-3*q)/(2*r))*5
L_z = (2*q/r)*5

print(f'L_z={L_z};   S_z={S_z}')
#%%
asym_peak = p/pp
print(f"Peak asymmetry: {asym_peak:.3f}")
# %%
#write code where the energy ranges are fixed but have random noise and then plot histogram and take mean+std as the values. 
#for less than half filled opposite sign than for the more than half filled one
-0.16396523/0.30300879
# %%
gy_xmcd_1 = np.loadtxt(r"C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\SS25\Magnetismus\xmcd Dy(1).txt")
gy_xmcd_2 = np.loadtxt(r"C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\SS25\Magnetismus\xmcd Dy(2).txt")
gy_xmcd_3 = np.loadtxt(r"C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\SS25\Magnetismus\xmcd Dy(3).txt")

gy_wl_1 = gy_xmcd_1[:,0]
gy_xmcd_1 = gy_xmcd_1[:,1]

gy_wl_2 = gy_xmcd_2[:,0]
gy_xmcd_2 = gy_xmcd_2[:,1]

gy_wl_3 = gy_xmcd_3[:,0]
gy_xmcd_3 = gy_xmcd_3[:,1]

all_gy_wl = [gy_wl_1,gy_wl_2,gy_wl_3]
all_gy_xmcd = [gy_xmcd_1,gy_xmcd_2,gy_xmcd_3]
labels = ['xmcd 1', 'xmcd 2', 'xmcd 3']

fig, axs = plt.subplots(1,3,figsize=(25,10),dpi=150)
axs = axs.flatten()
for i in range(3):
    ax = axs[i]
    plot = ax.plot(all_gy_wl[i], all_gy_xmcd[i], label=labels[i])
    ax.legend()
    ax.grid()


plt.tight_layout()
plt.show()
# %%
gy_xas_1 = np.loadtxt(r"C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\SS25\Magnetismus\xas Dy(1).txt")
gy_xas_2 = np.loadtxt(r"C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\SS25\Magnetismus\xas Dy(2).txt")
gy_xas_3 = np.loadtxt(r"C:\Users\Kyeez\OneDrive\Desktop\Uni\Master\SS25\Magnetismus\xas Dy(3).txt")

gy_wl_1 = gy_xas_1[:,0]
gy_xas_1 = gy_xas_1[:,1]

gy_wl_2 = gy_xas_2[:,0]
gy_xas_2 = gy_xas_2[:,1]

gy_wl_3 = gy_xas_3[:,0]
gy_xas_3 = gy_xas_3[:,1]

xas_gy_wl = [gy_wl_1,gy_wl_2,gy_wl_3]
xas_gy_xas = [gy_xas_1,gy_xas_2,gy_xas_3]
xas_labels = ['xas 1', 'xas 2', 'xas 3']

fig, axs = plt.subplots(1,3,figsize=(25,10),dpi=150)
axs = axs.flatten()
for i in range(3):
    ax = axs[i]
    plot = ax.plot(xas_gy_wl[i], xas_gy_xas[i], label=xas_labels[i])
    ax.legend()
    ax.grid()


plt.tight_layout()
plt.show()




# %%
figsize = (10,7)
dpi = 150 

plt.figure(figsize=figsize, dpi=dpi)
plt.plot(gy_wl,gy_xas, label='XAS', c='blue')
plt.plot(gy_wl,gy_xmcd, label='XMCD', c='orange')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=figsize, dpi=dpi)
plt.plot(gy_wl_2,gy_xas_2, label='XAS 2', c='blue')
plt.plot(gy_wl_2,gy_xmcd_2, label='XMCD 2', c='orange')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# %%
