#------------------------------------------------------------------------------------
# This program solves a nonlinear Stokes problem describing ice-shelf response to 
# sub-ice-shelf melting or freezing anomalies. The code relies on FEniCSx-see README
#------------------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
import numpy as np
import smb
from dolfinx.mesh import create_rectangle
from mesh_routine import get_surfaces, move_mesh
from mpi4py import MPI
from params import H, L, Nx, Nz, nt, t_f, t_r
from scipy.signal import convolve
from stokes import stokes_solve

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

# forcing functions (surface mass balance) are defined in smb module
smb_h = smb.smb_h
smb_s = smb.smb_s

t = np.linspace(0,t_f, nt)
# # Begin time stepping
for i in range(nt):

    print('Iteration '+str(i+1)+' out of '+str(nt))

    t_i = t[i]

    # Solve the Stoke problem for w = (u,p)
    w = stokes_solve(domain)
 
    # Move the mesh 
    domain = move_mesh(w,domain,t_i,smb_h,smb_s)
    h_i,s_i,x = get_surfaces(domain)

    h[:,i] = h_i
    s[:,i] = s_i
 

# Save scaled versions of the anomalies
np.savetxt(results_dir+'/dh',h/H-1)        # dh = upper surface anomaly
np.savetxt(results_dir+'/ds',s/H)          # ds = lower surface anomaly   
np.savetxt(results_dir+'/x',x/H)           # x = spatial coordinate 
np.savetxt(results_dir+'/t',t/t_r)         # t = time coordinate
np.savetxt(results_dir+'/t_r',[t_r])       # t_r = time scale 
np.savetxt(results_dir+'/H',[H])           # H = spatial scale


# Compute linear steady solution for verification via Green's function
# Green's function for steady, linearized problem:
G = (1./4)*np.pi*(1/np.cosh((np.pi*(x/H))/2.)**2) *(-3+np.pi*(x/H)*np.tanh((np.pi*x/H)/2.0))
# scaled basal melt rate:
M = smb_s(x,0)*t_r/H  
# compute convolution solution:
h0 = convolve(G,M,mode='same')*(x[1]-x[0])/H 

# plot the solutions:
inds = np.arange(0,nt,int(nt/100))
M_max = (t_r/H)*np.abs(smb_s(0,t)).max()
print('plotting...')
j=1
for i in inds:
    print('saving image '+str(j)+' out of '+str(inds.size))
    ex = 50     # vertical exaggeration factor
    plt.figure(figsize=(8,4))
    plt.title(r'$t=$'+'{:.2f}'.format(t[i]/t_r),fontsize=20)
    plt.plot(x/H,1+ex*h0,linewidth=2,color='k',linestyle='--',label=r'$1+\gamma h$ (steady linear)',zorder=41)
    plt.plot(x/H,1+ex*(h[:,i]-H)/H,linewidth=3,color='forestgreen',label=r'$1+\gamma h$ (nonlinear FEM)',zorder=31)
    plt.plot(x/H,ex*s[:,i]/H,linewidth=3,color='midnightblue',label=r'$\gamma s$ (nonlinear FEM)',zorder=32)
    plt.fill_between(x/H,y1=ex*s[:,i]/H, y2=1+ex*(h[:,i]-H)/H,facecolor='aliceblue',alpha=1.0)
    plt.fill_between(x/H,y1=-2*np.ones(np.size(x)), y2=ex*s[:,i]/H,facecolor='lightsteelblue',alpha=1,zorder=15)
    plt.ylabel(r'$z$',fontsize=16)
    plt.xlabel(r'$x$',fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.annotate(xy=(-0.5*L/H+0.2,1.075),text='air',fontsize=16,zorder=50)
    plt.annotate(xy=(-0.5*L/H+0.2,0.075),text='ice',fontsize=16,zorder=51)
    plt.annotate(xy=(-0.5*L/H+0.2,-0.125),text='water',fontsize=16,zorder=52)
    plt.legend(fontsize=16,ncol=1,bbox_to_anchor=(0.8,-0.5))
    plt.ylim(-0.4,1.2)
    plt.xlim(-0.5*L/H,0.5*L/H)

    axins = plt.gca().inset_axes([0.0, -0.45, 1.0, 0.25],sharex=plt.gca())
    axins.plot(x/H,smb_s(x,t[i])*t_r/H,linewidth=2,color='crimson',linestyle='-')
    axins.set_ylabel(r'$m$',fontsize=16,color='crimson')
    axins.set_ylim(-np.abs(M).max()-1e-4,np.abs(M).max()+1e-4)
    axins.tick_params(axis='y', which='major', labelsize=16,color='crimson',labelcolor='crimson')
    axins.tick_params(axis='x',direction="in",which='major', labelsize=16,top=True, labeltop=True, bottom=False, labelbottom=False)
    axins.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4),useMathText=True)
    plt.setp(axins.get_xticklabels(), visible=False)
    plt.savefig(results_dir+'/pngs/fig_'+str(j),bbox_inches='tight')
    plt.close()
    j+=1