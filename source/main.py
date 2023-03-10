#------------------------------------------------------------------------------------
# This program solves a nonlinear Stokes problem describing sub-ice-shelf melt/freeze
#------------------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
import numpy as np
from dolfinx.mesh import create_rectangle
from mesh_routine import get_surfaces, move_mesh
from mpi4py import MPI
from params import H, L, Nx, Nz, dt, nt, t_f, t_r, z_max
from scipy.signal import convolve
from scipy.special import erf
from stokes import stokes_solve
from ufl import exp, sign

results_dir = '../results'
if os.path.isdir(results_dir)==False:
    os.mkdir(results_dir)   # Make a directory for the results.
if os.path.isdir(results_dir+'/pngs')==False:
    os.mkdir(results_dir+'/pngs')   # Make a directory for the results.

# Load mesh
p0 = [-L/2.0,0.0]
p1 = [L/2.0,H]
domain = create_rectangle(MPI.COMM_WORLD,[p0,p1], [Nx, Nz])

# Define arrays for saving surfaces
h_i,s_i,x = get_surfaces(domain)
nx = x.size
h = np.zeros((nx,nt))
s = np.zeros((nx,nt))

# Define forcing function
def smb(x,z):
    # Surface mass balance functions (at upper and basal surfaces)
    # The sign(z-z_max) terms are used to ensure that the functionss
    # only apply at the appropriate surfaces; 
    # z_max is the maximum channel height allowed in the model (see params.py)
    m0 = 5                  # max basal melt rate (m/yr)
    stdev = 10*H/3          # standard deviation for Gaussian basal melt anomaly
    a0 = m0*np.sqrt(np.pi)*stdev*erf(L/(2*stdev)) / L
    m = m0*(exp(-x**2/(stdev**2))/3.154e7)*0.5*(1-sign(z-z_max))
    a = a0*0.5*(sign(z-z_max)+1)/3.154e7
    return m+a

# xdmf = io.XDMFFile(domain.comm, results_dir+"/stokes.xdmf", "w")
# xdmf.write_mesh(domain)

t = 0                       # time
# # Begin time stepping
for i in range(nt):

    print('Iteration '+str(i+1)+' out of '+str(nt))

    # Solve the Stoke problem for w = (u,p)
    w = stokes_solve(domain)

    # xdmf.write_function(w.sub(0).sub(1), t)
   
    #xdmf.write_function(w.sub(0).sub(0), t)
 
    # Move the mesh 
    domain = move_mesh(w,smb,domain)
    h_i,s_i,x = get_surfaces(domain)

    h[:,i] = h_i
    s[:,i] = s_i
 
    # Update time
    t += dt

# xdmf.close()

# Compute linear steady solution for validation
m0 = (5/3.154e7)*t_r/H # max basal melt rate (m/yr)
stdev = 10/3           # standard deviation for Gaussian basal melt anomaly
x0 = x/H
# Green function for linearized problem
G = (1./4)*np.pi*(1/np.cosh((np.pi*x0)/2.)**2) *(-3+np.pi*x0 *np.tanh((np.pi* x0)/2.0))
M = m0*np.exp(-x0**2/(stdev**2))
dx = x0[1]-x0[0]
h0 = dx*convolve(G,M,mode='same')

t_ = np.linspace(0,t_f, nt)

inds = np.arange(0,nt,int(nt/100))

print('plotting...')
j=1
for i in inds:
    print('saving image '+str(j)+' out of '+str(inds.size))
    ex = 50     # vertical exaggeration factor
    plt.figure(figsize=(8,6))
    plt.title(r'$t\, / \, t_\mathrm{r}=$'+'{:.2f}'.format(t_[i]/t_r),fontsize=20)
    plt.plot(x/H,1+ex*h0,linewidth=2,color='k',linestyle='--',label=r'$1+\gamma h$ (steady linear)',zorder=41)
    plt.plot(x/H,1+ex*(h[:,i]-H)/H,linewidth=3,color='forestgreen',label=r'$1+\gamma h$ (nonlinear FEM)',zorder=31)
    plt.plot(x/H,ex*s[:,i]/H,linewidth=3,color='midnightblue',label=r'$1+\gamma s$ (nonlinear FEM)',zorder=32)
    plt.fill_between(x/H,y1=ex*s[:,i]/H, y2=1+ex*(h[:,i]-H)/H,facecolor='aliceblue',alpha=1.0)
    plt.fill_between(x/H,y1=-2*np.ones(np.size(x)), y2=ex*s[:,i]/H,facecolor='lightsteelblue',alpha=1,zorder=15)
    plt.xlabel(r'$x\,/\, H$',fontsize=16)
    plt.ylabel(r'$z\,/\, H$',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16,ncol=1,bbox_to_anchor=(0.8,-0.11))
    plt.ylim(-0.2,1.2)
    plt.xlim(-0.5*L/H,0.5*L/H)
    plt.savefig(results_dir+'/pngs/fig_'+str(j),bbox_inches='tight')
    plt.close()
    j+=1

np.savetxt(results_dir+'/h',h)           # h = upper surface
np.savetxt(results_dir+'/s',s)           # s = lower surface   
np.savetxt(results_dir+'/x',x)           # x = spatial coordinate 
np.savetxt(results_dir+'/t',t_/t_r)      # t = time coordinate
