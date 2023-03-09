#------------------------------------------------------------------------------
# These functions are used to:
# (1) update the mesh at each timestep by solving the
#     surface kinematic equations, AND...
# (2) compute the grounding line positions.
#------------------------------------------------------------------------------

from params import dt
from dolfinx import *
from ufl import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from bdry_conds import WaterBoundary,TopBoundary
from petsc4py.PETSc import ScalarType

# ------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def move_mesh(w,smb,domain):
    # this function computes the surface displacements and moves the mesh
    # by solving Laplace's equation for a smooth displacement function
    # defined for all mesh vertices

    V = fem.FunctionSpace(domain, ("CG", 1))
    x = SpatialCoordinate(domain)


    # displacement at upper and lower boundaries
    disp_ex = w.sub(0).sub(1) - w.sub(0).sub(0)*Dx(x[1],0)+smb(x[0],x[1])

    disp_bdry = fem.Function(V)
    
    disp_bdry.interpolate(fem.Expression(disp_ex, V.element.interpolation_points()))

    dofs_1 = fem.locate_dofs_geometrical(V, WaterBoundary)
    dofs_2 = fem.locate_dofs_geometrical(V, TopBoundary)

    # # define displacement boundary conditions on upper and lower surfaces
    bc1 = fem.dirichletbc(disp_bdry, dofs_1)
    bc2 = fem.dirichletbc(disp_bdry, dofs_2)

    bcs = [bc1,bc2]

    # # solve Laplace's equation for a smooth displacement field on all vertices,
    # # given the boundary displacement disp_bdry
    disp = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(disp), grad(v))*dx
    f = fem.Constant(domain, ScalarType(0.0))
    L = f*v*dx

    problem = fem.petsc.LinearProblem(a, L, bcs=bcs)
    disp = problem.solve()

    disp_vv = disp.x.array

    M = domain.geometry.x

    M[:,1] += dt*disp_vv

    return domain


def get_surfaces(domain):

    M = domain.geometry.x

    N = np.size(M[:,0])

    x = np.sort(M[:,0])
    z = M[:,1][np.argsort(M[:,0])]
    x_u = np.unique(M[:,0])
    h = np.zeros(x_u.size)
    s = np.zeros(x_u.size)

    for i in range(h.size):
        h[i] = np.max(z[np.where(np.isclose(x_u[i],x))])
        s[i] = np.min(z[np.where(np.isclose(x_u[i],x))])

    return h,s,x_u