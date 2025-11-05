#%%
import numpy as np 
import matplotlib.pyplot as plt

def Iterator_retarded(energy:float, eps:np.ndarray(4), t_matrix:np.ndarray(4), eta:float) -> np.ndarray(4,dtype=complex): # type: ignore
    '''
    Iteration that returns the Gs(retarded surface Green-function) and Gb(retarded bulk Green-function) as a complex 4x4 matrix
    -energy: Onsite energy 
    -eps: onsite Hamiltonian
    -t_matrix: hoppping matrix
    '''
    z = (energy + 1j*eta) * np.eye(4, dtype=np.complex128)   
    alpha = t_matrix
    beta =  np.transpose(np.conjugate(t_matrix))
    Epsilon_surf = eps
    Epsilon_bulk = eps

    for i in range(500):
        aux = np.linalg.inv((z - Epsilon_bulk))
        Epsilon_surf = Epsilon_surf + alpha@aux@beta
        Epsilon_bulk = Epsilon_bulk + alpha@aux@beta + beta@aux@alpha
        alpha = alpha@aux@alpha
        beta = beta@aux@beta
        if np.linalg.norm(alpha) < 1e-10 :
            break
    Gs = np.linalg.inv((z - Epsilon_surf))
    Gb = np.linalg.inv((z - Epsilon_bulk))

    return Gs,Gb

def Iterator_advanced(energy:float, eps:np.ndarray(4), t_matrix:np.ndarray(4), eta:float) -> np.ndarray(4,dtype=complex): # type: ignore
    '''
    Iteration that returns the Gs(advanced surface Green-function) and Gb(advanced bulk Green-function) as a complex 4x4 matrix
    -energy: Onsite energy 
    -eps: onsite Hamiltonian
    -t_matrix: hoppping matrix
    '''
    z = (energy - 1j*eta) * np.eye(4)   
    alpha = t_matrix
    beta =  np.transpose(np.conjugate(t_matrix))
    Epsilon_surf = eps
    Epsilon_bulk = eps

    for i in range(500):
        aux = np.linalg.inv((z - Epsilon_bulk))
        Epsilon_surf = Epsilon_surf + alpha@aux@beta
        Epsilon_bulk = Epsilon_bulk + alpha@aux@beta + beta@aux@alpha
        alpha = alpha@aux@alpha
        beta = beta@aux@beta
        if np.linalg.norm(alpha) < 1e-10:
            break
    Gs = np.linalg.inv((z - Epsilon_surf))
    Gb = np.linalg.inv((z - Epsilon_bulk))

    return Gs,Gb


def onsite_matrix(t:float, mu:float, h:float, delta:float):
    '''    up, down, down^dagger, up^dagger
    Returns the matrix [[2*t - mu-h, 0, delta, 0], [0, 2*t - mu+h, 0, -delta], [delta, 0, -2*t + mu-h,0], [0, -delta, 0, -2*t + mu + h]]
    alpha: Rashba soc
    h: Zeeman energy 
    onsite: 2t - mu
    delta: SC order parameter
    t: hopping parameter
    mu: chemical potential
    '''
    matrix = np.array([[2*t - mu-h, 0, delta, 0], [0, 2*t - mu+h, 0, -delta], [delta, 0, -2*t + mu-h,0], [0, -delta, 0, -2*t + mu + h]])
    return matrix

def t_matrix(t:float, alpha:float):
    '''
    Returns the matrix [[-t, alpha, 0, 0],[-alpha, -t, 0, 0],[0, 0, t, alpha],[0, 0, -alpha, t]]
    t: hopping parameter
    alpha: rashba soc 
    '''
    matrix = np.array([[-t, alpha, 0, 0],[-alpha, -t, 0, 0],[0, 0, t, alpha],[0, 0, -alpha, t]])
    return matrix

def plot_ldos_chain(energy_array:np.array, surface_green_function:list,bulk_green_function:list, xlim:float):
    '''
    Plot the LDOS of surface bulk Green-Function
    enegry_array: the energies you want to plot over
    surface_green_function: surface green function as list of 4x4 arrays
    bulk_green_function: bulk green function as list of 4x4 arrays
    xlim: bounds of x-axis
    '''

    plt.figure(figsize=(12,8))
    plt.plot(energy_array, [(-1/np.pi)*np.imag(np.trace(Gb_val)) for Gb_val in bulk_green_function])
    plt.gca().set_facecolor('lightyellow')
    plt.xlabel('Energy')
    plt.ylabel('DOS')
    plt.xlim([-xlim,xlim])
    plt.title('Bulk DOS')
    plt.grid()
    plt.show()

    '''Plot the LDOS of  surface Green-Function'''
    plt.figure(figsize=(12,8))
    #plt.plot(energy_array, [(Gs_imag[1, 1])  for Gs_imag in Gs_imag_list], label='Holes', c='red')
    #plt.plot(energy_array,  [(t*Gs_imag[0, 0])  for Gs_imag in Gs_imag_list], label='Electrons', c='blue')
    plt.plot(energy_array, [(-1/np.pi)*np.imag(np.trace(Gs_val)) for Gs_val in surface_green_function])
    plt.xlabel('Energy')
    plt.ylabel('DOS')
    plt.xlim([-xlim,xlim])
    plt.gca().set_facecolor('lightyellow')
    plt.title('Surface DOS')
    plt.grid()
    plt.show()
    
def calc_orbital_functions(h_array:np.array, energy_array:np.array,t:float, mu:float, delta:float, alpha:float ):
    '''
    alpha: Rashba soc
    onsite: 2t - mu
    delta: SC order parameter
    t: hopping parameter
    mu: chemical potential
    h_array: 1d array to give values for h to plot over
    energy_array: 1d array to give values for e
    '''
    gs_uu_r = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
    gs_dd_r = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
    gs_du_r = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
    gs_ud_r = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)

    gs_uu_a = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
    gs_dd_a = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
    gs_du_a = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)
    gs_ud_a = np.zeros((len(h_array), len(energy_array)), dtype=np.complex128)

    for i,h in enumerate(h_array):
        for j,e in enumerate(energy_array):
            Gs_r, Gb_r = Iterator_retarded(e, onsite_matrix(t, mu, h, delta), t_matrix(t, alpha))
            Gs_a, Gb_a = Iterator_advanced(e, onsite_matrix(t, mu, h, delta), t_matrix(t, alpha))
            
            gs_r = Gs_r[0:2,2:4]
            gs_a = Gs_a[0:2,2:4]
            
            gs_uu_r[i, j] = gs_r[0,1]
            gs_ud_r[i, j] = gs_r[0,0]
            gs_du_r[i, j] = gs_r[1,1]
            gs_dd_r[i, j] = gs_r[1,0]
            
            gs_uu_a[i, j] = gs_a[0,1]
            gs_ud_a[i, j] = gs_a[0,0]
            gs_du_a[i, j] = gs_a[1,1]
            gs_dd_a[i, j] = gs_a[1,0]


    d0 = .5*((gs_ud_a - gs_ud_r) - (gs_du_a - gs_du_r))
    dx = -.5*((gs_uu_a - gs_uu_r) - (gs_dd_a - gs_dd_r))
    dy = -.5j*((gs_uu_a - gs_uu_r) + (gs_dd_a - gs_dd_r))
    dz = .5*((gs_ud_a - gs_ud_r) + (gs_du_a - gs_du_r))
    
    
    
def my_color_plot(x:np.ndarray,y:np.ndarray,z:np.ndarray,
                  cmap:str,zmin:float=None,zmax:float=None,
                  title:str=None,xaxis:str=None,yaxis:str=None,cbar:str=None,
                  xline:float=None,yline:float=None,box:bool=True,
                  data_display : dict = None):
    xm,ym = np.meshgrid(x,y)
    if zmin is not None and zmax is not None:
        norm = plt.Normalize(zmin,zmax)
        zc = np.clip(z,zmin,zmax)
    else:
        norm = None
        zc = z
    
    fig,ax = plt.subplots()
    c = ax.contourf(xm,ym,zc,levels=300,cmap=cmap,norm = norm)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if cbar:
        plt.colorbar(c, label=cbar)
    plt.title(title)
    if yline:
        plt.axhline(y=yline,color='k',linestyle='--')
    if xline:
        plt.axvline(x=xline,color='k',linestyle='--')
    if box and data_display:
        textstr = '\n'.join([f'{key} = {value:.2f}' for key, value in data_display.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

#%%