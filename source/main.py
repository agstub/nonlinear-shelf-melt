#------------------------------------------------------------------------------------
# This program solves a nonlinear Stokes problem describing sub-ice-shelf melt/freeze
#------------------------------------------------------------------------------------
from dolfinx import *
from ufl import exp,sign
from mpi4py import MPI
import numpy as np
from stokes import stokes_solve
from mesh_routine import move_mesh,get_surfaces
import os
from params import L,H,nt,dt,Nx,Nz,z_max,t_r,t_f,rho_w,rho_i
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.signal import convolve

results_dir = '../results'
if os.path.isdir(results_dir)==False:
    os.mkdir(results_dir)   # Make a directory for the results.
if os.path.isdir(results_dir+'/pngs')==False:
    os.mkdir(results_dir+'/pngs')   # Make a directory for the results.

# Load mesh
p0 = [-L/2.0,0.0]
p1 = [L/2.0,H]
domain = mesh.create_rectangle(MPI.COMM_WORLD,[p0,p1], [Nx, Nz])

# Define arrays for saving surfaces
h_i,s_i,x = get_surfaces(domain)
nx = x.size
h = np.zeros((nx,nt))
s = np.zeros((nx,nt))

# Define forcing function
def smb(x,z):
    # Surface mass balance (upper and basal surfaces)
    m0 = 5                  # max basal melt rate (m/yr)
    stdev = 10*H/3          # standard deviation for Gaussian basal melt anomaly
    a0 = m0*np.sqrt(np.pi)*stdev*erf(L/(2*stdev)) / L
    f = m0*(exp(-x**2/(stdev**2))/3.154e7)*0.5*(1-sign(z-z_max))
    f += a0*0.5*(sign(z-z_max)+1)/3.154e7
    return f

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

# # Compute linear steady solution for validation
# Green function for linearized problem
m0 = (5/3.154e7)*t_r/H # max basal melt rate (m/yr)
stdev = 10/3           # standard deviation for Gaussian basal melt anomaly
x0 = x/H
G = (1./4)*np.pi*(1/np.cosh((np.pi*x0)/2.)**2) *(-3+np.pi*x0 *np.tanh((np.pi* x0)/2.0))
M = m0*np.exp(-x0**2/(stdev**2))
dx = x0[1]-x0[0]
h0 = dx*convolve(G,M,mode='same')

t_ = np.linspace(0,t_f,nt)

print('plotting...')
for i in range(nt):
    print('saving image '+str(i+1)+' out of '+str(nt))
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
    plt.savefig(results_dir+'/pngs/fig_'+str(i),bbox_inches='tight')
    plt.close()

np.savetxt(results_dir+'/h',h)           # h = upper surface
np.savetxt(results_dir+'/s',s)           # s = lower surface   
np.savetxt(results_dir+'/x',x)           # x = spatial coordinate 
np.savetxt(results_dir+'/t',t_/t_r)      # t = time coordinate
