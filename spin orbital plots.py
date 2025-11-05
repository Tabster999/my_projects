#%%
import numpy as np 
import matplotlib.pyplot as plt 

#%%
'Define functions to calculate the orbitals'


def calc_d0(k,h,m,alpha,Delta,mu,Energy):
    d0 = 1/(h**4 + 2* h**2 * ((alpha*k)**2 - Delta**2 - ((k**2/(2*m)) - mu)**2 - Energy**2) + (((alpha*k + (k**2/(2*m)) - mu))**2 + (Delta - Energy)*(Delta + Energy))*((alpha*k - (k**2/(2*m)) + mu)**2 + (Delta - Energy)*(Delta + Energy))
) *  ((alpha*k)**2+Delta**2+((k**2/(2*m)) - mu)**2-h**2-Energy**2)*Delta
    return d0
def calc_dx(k,h,m,alpha,Delta,mu,Energy):
    dx = (1/(h**4 + 2* h**2 * ((alpha*k)**2 - Delta**2 - ((k**2/(2*m)) - mu)**2 - Energy**2) + ((alpha*k + (k**2/(2*m)) - mu)**2 + (Delta - Energy)*(Delta + Energy))*((alpha*k - (k**2/(2*m)) + mu)**2 + (Delta - Energy)*(Delta + Energy)
 ))) *  -2j*alpha*k*h*Delta
    return dx
def calc_dy(k,h,m,alpha,Delta,mu,Energy):
    dy = 1/(h**4 + 2* h**2 * ((alpha*k)**2 - Delta**2 - ((k**2/(2*m)) - mu)**2 - Energy**2) + ((alpha*k + (k**2/(2*m)) - mu)**2 + (Delta - Energy)*(Delta + Energy))*((alpha*k - (k**2/(2*m)) + mu)**2 + (Delta - Energy)*(Delta + Energy))
) *  -2*alpha*k*((k**2/(2*m)) - mu)*Delta
    return dy
def calc_dz(k,h,m,alpha,Delta,mu,Energy):
    dz = 1/(h**4 + 2* h**2 * ((alpha*k)**2 - Delta**2 - ((k**2/(2*m)) - mu)**2 - Energy**2) + ((alpha*k + (k**2/(2*m)) - mu)**2 + (Delta - Energy)*(Delta + Energy))*((alpha*k - (k**2/(2*m)) + mu)**2 + (Delta - Energy)*(Delta + Energy))
) *  2*h*Energy*Delta
    return dz
h_values = np.linspace(0,2,200)
k = 1
m = 1
Delta = 1
mu = .025*Delta
alpha = .2*Delta
Energy = .12*Delta


# %%
'''Create lists to plot the orbital functions'''
d0_list = []
dx_list = []
dy_list = []
dz_list = []

for h in h_values:
    d0_entry = calc_d0(k,h,m,alpha,Delta,mu,Energy)
    dx_entry = calc_dx(k,h,m,alpha,Delta,mu,Energy)
    dy_entry = calc_dy(k,h,m,alpha,Delta,mu,Energy)
    dz_entry = calc_dz(k,h,m,alpha,Delta,mu,Energy)
    
    d0_list.append(d0_entry)
    dx_list.append(dx_entry)
    dy_list.append(dy_entry)
    dz_list.append(dz_entry)

d_list = []
title_str = ['d0 real','dx_im','dy real ','dz real']
d_list = [d0_list,dx_list,dy_list,dz_list]

# %%
'''Plot for d0 and dz'''
fig,ax = plt.subplots()
plt.plot(h_values,np.array( d0_list).real,c='r',label='d0 real')
plt.plot(h_values, np.array(dz_list).real, c='b',label='dz real')
textstr = '\n'.join((
    r'$\mu/\Delta = %.3f$'%(mu,),
    r'$k = %.2f$'%(k,),
    r'$\alpha/\Delta = %.2f  $' %(alpha,),
    r'$E/\Delta = %.2f$' %(Energy,)
))
props = dict(boxstyle='round',facecolor='lightyellow',alpha=1)
ax.text(0.75,.95,textstr,transform=ax.transAxes,fontsize=14,verticalalignment='top',bbox=props)
plt.axvline(x=np.sqrt(mu**2 + Delta**2),color='k',linestyle='--',label='phase transition')
plt.legend(loc=4)
plt.tight_layout()
plt.gca().set_facecolor('lightgray')
plt.grid()
plt.show()

'Plots for dx and dy'
fig2,ax2 = plt.subplots()
plt.plot(h_values, np.array(dx_list).imag, c='darkgreen',label='dx imag')
plt.plot(h_values, np.array(dy_list).real, c='purple', linestyle='dashdot', label='dy real')
textstr2 = '\n'.join((
    r'$\mu/\Delta = %.3f$'%(mu,),
    r'$k = %.2f$'%(k,),
    r'$\alpha/\Delta = %.2f  $' %(alpha,),
    r'$E/\Delta = %.2f$' %(Energy,)
))
props2 = dict(boxstyle='round',facecolor='lightyellow',alpha=1)
ax2.text(0.75,.3,textstr2,transform=ax2.transAxes,fontsize=14,verticalalignment='top',bbox=props)
plt.axvline(x=np.sqrt(mu**2 + Delta**2),color='k',linestyle='--',label='phase transition')
plt.legend()
plt.grid()
plt.gca().set_facecolor('lightgray')
plt.figure()
plt.plot(h_values, (np.array(dx_list) + 1j*np.array(dy_list)).real, label='dx+idy real')
plt.plot(h_values, (np.array(dx_list) - 1j*np.array(dy_list)).real, label='dx -idy real')
plt.legend()
plt.show()
plt.figure()
plt.plot(h_values, np.abs((np.array(dx_list) + 1j*np.array(dy_list)).imag), label='dx+idy imag')
plt.plot(h_values, (np.array(dx_list) - 1j*np.array(dy_list)).imag,label='dx-idy imag')
plt.legend()
plt.show()

# %%
