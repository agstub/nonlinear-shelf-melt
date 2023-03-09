# This file contains the functions needed for solving the upper-convected Maxwell problem.
from params import H,B,rho_i,rm2,g,rho_w,eps_v,sea_level,dt
from bdry_conds import mark_boundary,LeftBoundary,RightBoundary,TopBoundary
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import *
from ufl import *


def eta(u):
      # ice viscosity 
      return 0.5*B*((inner(sym(grad(u)),sym(grad(u)))+eps_v)**(rm2/2.0))

def weak_form(u,p,v,q,f,g_base,ds,nu):
    # Weak form residual of the ice-shelf problem
    F = 2*eta(u)*inner(sym(grad(u)),sym(grad(v)))*dx
    F += (- div(v)*p + q*div(u))*dx - inner(f, v)*dx
    F += (g_base+rho_w*g*dt*inner(u,nu))*inner(v,nu)*ds(3)
    return F

def stokes_solve(domain):
        # Stokes solver for the ice-shelf problem using Taylor-Hood elements

        # Define function spaces
        P1 = FiniteElement('P',triangle,1)     # Pressure p
        P2 = FiniteElement('P',triangle,2)     # Velocity u
        element = MixedElement([[P2,P2],P1])
        W = fem.FunctionSpace(domain,element)  # Function space for (u,p)

        #---------------------Define variational problem------------------------
        w = fem.Function(W)
        (u,p) = split(w)
        (v,q) = TestFunctions(W)
      
        # Neumann condition at ice-water boundary
        x = SpatialCoordinate(domain)
        g_0 = rho_w*g*(sea_level-x[1])
        g_base = 0.5*(g_0+abs(g_0))

        # Body force
        f = fem.Constant(domain,PETSc.ScalarType((0,-rho_i*g)))      

        # Outward-pointing unit normal to the boundary  
        nu = FacetNormal(domain)           

        # Mark bounadries of mesh and define a measure for integration
        facet_tag = mark_boundary(domain)
        ds = Measure('ds', domain=domain, subdomain_data=facet_tag)

        # Define boundary conditions on the inflow/outflow boundary
        facets_1 = mesh.locate_entities_boundary(domain, domain.topology.dim-1, LeftBoundary)        
        facets_2 = mesh.locate_entities_boundary(domain, domain.topology.dim-1, RightBoundary)
        dofs_1 = fem.locate_dofs_topological(W.sub(0).sub(0), domain.topology.dim-1, facets_1)
        dofs_2 = fem.locate_dofs_topological(W.sub(0).sub(0), domain.topology.dim-1, facets_2)
        bc1 = fem.dirichletbc(PETSc.ScalarType(0), dofs_1,W.sub(0).sub(0))
        bc2 = fem.dirichletbc(PETSc.ScalarType(0), dofs_2,W.sub(0).sub(0))
        bcs = [bc1,bc2]

        # Define weak form
        F = weak_form(u,p,v,q,f,g_base,ds,nu)

        # Solve for (u,p)
        problem = fem.petsc.NonlinearProblem(F, w, bcs=bcs)
        solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

        log.set_log_level(log.LogLevel.WARNING)
        n, converged = solver.solve(w)
        assert(converged)
      
        return w