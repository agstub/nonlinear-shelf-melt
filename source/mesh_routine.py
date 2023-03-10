#-------------------------------------------------------------------------------------
# These functions are used to:
# (1) move the mesh at each timestep according the solution and forcings (move_mesh), 
# (2) retrieve numpy arrays of the upper and lower surface elevations (get_surfaces)
#-------------------------------------------------------------------------------------

import numpy as np
from bdry_conds import TopBoundary, WaterBoundary
from dolfinx.fem import (Constant, Expression, Function, FunctionSpace,
                         dirichletbc, locate_dofs_geometrical)
from dolfinx.fem.petsc import LinearProblem
from params import dt
from petsc4py.PETSc import ScalarType
from ufl import (Dx, SpatialCoordinate, TestFunction, TrialFunction, dx, grad,
                 inner)


# ------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
def move_mesh(w,domain,t,smb_h,smb_s):
    # this function computes the surface displacements and moves the mesh
    # by solving Laplace's equation for a smooth displacement function
    # defined for all mesh vertices

    V = FunctionSpace(domain, ("CG", 1))
    x = SpatialCoordinate(domain)

    # displacement at upper and lower boundaries
    disp_h = w.sub(0).sub(1) - w.sub(0).sub(0)*Dx(x[1],0)+smb_h(x[0],t)
    disp_s = w.sub(0).sub(1) - w.sub(0).sub(0)*Dx(x[1],0)+smb_s(x[0],t)

    disp_h_fcn = Function(V)
    disp_s_fcn = Function(V)
    
    disp_h_fcn.interpolate(Expression(disp_h, V.element.interpolation_points()))
    disp_s_fcn.interpolate(Expression(disp_s, V.element.interpolation_points()))

    dofs_1 = locate_dofs_geometrical(V, WaterBoundary)
    dofs_2 = locate_dofs_geometrical(V, TopBoundary)

    # # define displacement boundary conditions on upper and lower surfaces
    bc1 = dirichletbc(disp_s_fcn, dofs_1)
    bc2 = dirichletbc(disp_h_fcn, dofs_2)

    bcs = [bc1,bc2]

    # # solve Laplace's equation for a smooth displacement field on all vertices,
    # # given the boundary displacement disp_bdry
    disp = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(disp), grad(v))*dx
    f = Constant(domain, ScalarType(0.0))
    L = f*v*dx

    problem = LinearProblem(a, L, bcs=bcs)
    disp = problem.solve()

    disp_vv = disp.x.array

    X = domain.geometry.x

    X[:,1] += dt*disp_vv

    return domain


def get_surfaces(domain):
# retrieve numpy arrays of the upper and lower surface elevations
    X = domain.geometry.x
    x = np.sort(X[:,0])
    z = X[:,1][np.argsort(X[:,0])]
    x_u = np.unique(X[:,0])
    h = np.zeros(x_u.size)      # upper surface elevation
    s = np.zeros(x_u.size)      # lower surface elevation

    for i in range(h.size):
        h[i] = np.max(z[np.where(np.isclose(x_u[i],x))])
        s[i] = np.min(z[np.where(np.isclose(x_u[i],x))])

    return h,s,x_u