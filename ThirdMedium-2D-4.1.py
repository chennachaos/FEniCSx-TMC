# this FenicsX code is intended to implement the third medium contact approach described in the paper "A third medium approach for contact using first and second order finite elements" by Wriggers et al. (2025), specifically aiming to reproduce the example from Section 4.1 ("Self-contact within a box").

# import libraries and modules ========================================

import importlib.util
# The module 'importlib.util' provides functions to interact with Python's import system

# For MPI-based parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# The library 'mpi4py' provides bindings for the Message Passing Interface (MPI), which is used for parallel programming.
# Note that it is important to first from mpi4py import MPI to ensure that MPI is correctly initialised.

import numpy as np
# numpy is a library for numerical computations in Python
# This line imports it under the alias np

# PETSc solvers
from petsc4py import PETSc

# specific functions from dolfinx modules
from dolfinx import fem, io, mesh, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# specific functions from ufl modules
import ufl
from ufl import grad, inner, derivative, dot, inv, tr, det, TestFunctions, TrialFunction, TestFunction
# basix finite elements
from basix.ufl import element, mixed_element, quadrature_element

# create mesh and function space ======================================

# To align with Chenna's workflow, it is recommended to setup the geometry model using GMSH. 

# load GMSH file
# msh, markers, facet_tags = io.gmshio.read_from_msh("Beam-C3DQ1.msh", MPI.COMM_WORLD)
msh, cell_tags, facet_tags = io.gmshio.read_from_msh("thirdmedium-box-ex1-Q2-mesh1.msh", MPI.COMM_WORLD, 0, 2)
# 0 1 "pointA"
# 0 2 "pointB"
# 0 3 "pointC"
# 2 4 "solid"
# 2 5 "medium"

print(f"facet_tags: {facet_tags.values}")  #TODO: why facet tag empty?

gdim = msh.topology.dim
fdim = gdim - 1

# extract coordinate of mesh
x = ufl.SpatialCoordinate(msh)

# read physical ID
PointID_A = 1
PointID_B = 2
PointID_C = 3
VolID_solid = 4
VolID_medium = 5

# Define element degree
degree_u = 2
degree_p = degree_u - 1 

# Create element, TODO: which degree should be used for p and q? they are gradients of displacement.
element_disp = element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
# element_p = element("Lagrange", msh.topology.cell_name(), degree_u) 
# element_q = element("Lagrange", msh.topology.cell_name(), degree_u) 
# element_p = element("DG", msh.topology.cell_name(), degree_p)
# element_q = element("DG", msh.topology.cell_name(), degree_p)
element_lower = element("DG", msh.topology.cell_name(), 0)

# create a function space V on the mesh
V = fem.functionspace(msh, element_disp) 

# defining the trial(unknown) and test(small variation) function
u = fem.Function(V)

u_test = TestFunction(V) # test function for displacement

# define kinematic quantities =========================================

# Spatial dimension
dimTrial = len(u)

# Identity tensor
I = ufl.variable(ufl.Identity(dimTrial))
# Deformation gradient
F = ufl.variable(I + ufl.grad(u))
Finv = ufl.variable(ufl.inv(F))
# Right Cauchy-Green tensor
C = ufl.variable(F.T * F)

# Invariants of deformation tensors
Ic = ufl.variable(tr(C))
J = ufl.variable(det(F))

# Cinv = ufl.variable(inv(C))
# Fbar = ufl.variable(J**(-1.0/3.0)*F)
# Cbar = ufl.variable(Fbar.T*Fbar)
# CbarInv = ufl.variable(inv(Cbar))
# ICbar = ufl.variable(tr(Cbar))

# the rotation angle phi, in the case of 2D, make sure phi never goes to 0 
eps_ = fem.Constant(msh, PETSc.ScalarType(1.0e-12))
phi = (F[0,1]-F[1,0])/(F[0,0]+F[1,1] + eps_)

# define constitutive model ===========================================
# Define the elasticity model via a strain energy density function $\Psi$, and create the expression for the first Piola-Kirchhoff stress.

# Elasticity parameters
# Em = fem.Constant(msh, PETSc.ScalarType(100)) # MPa
# Nu = fem.Constant(msh, PETSc.ScalarType(0.4))
# shearmod = Em/(2*(1 + Nu))
# bulkmod = Em/3.0/(1.0-2*Nu)
# lamda1 = bulkmod - 2.0*shearmod/3.0

bulkmod = fem.Constant(msh, PETSc.ScalarType(20.0)) # K = 20, from Section 4.1
shearmod = fem.Constant(msh, PETSc.ScalarType(10.0)) # Mu = 10.0, from Section 4.1
# the scaling factor for the strain energy of the third medium
Gamma = fem.Constant(msh, PETSc.ScalarType(2.0e-2)) # gamma = 2.0e-4, from Section 4.1
# the penalty parameter which enforces the regularization
alpha_r = fem.Constant(msh, PETSc.ScalarType(0.1)) # alpha_r=100 from Section 4.1
# TODO: I cannot get converged results according to the Paper's setting. Note that the results are sensitive to the value of alpha_r and Gamma. 

# strain energy density (compressible/incompressible neo-Hookean model)
# Psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
# Psi  = 0.5*mu_active*(ICbar-3) + p*(J-1)

# elastomer solid in Wriggers's CMAME paper 2025
Psi_solid = (bulkmod/2) * (ufl.ln(J))**2 + (shearmod/2) * (J**(-2.0/3.0) * Ic - 2)
# third medium, in the case of 2D. 
# the regularization term \nabla J can be omitted when using finite elements with linear shape functions.
W_Phi_J = 0.5*Gamma*alpha_r*(inner(ufl.grad(phi), ufl.grad(phi)) + inner(ufl.grad(J), ufl.grad(J)))
#Psi_medium = Gamma*( (shearmod/2.0) * (J**(-2.0/3.0) * Ic - 2) ) + W_Phi_J  # keep consistent with paper, by omitting the first term in Psi_solid
Psi_medium = Gamma*( (shearmod/2.0) * (J**(-2.0/3.0) * Ic - 2) + (bulkmod/2) * (ufl.ln(J))**2 ) + W_Phi_J  # With the volumetric term

# automatically calculated PK1 Stress for hyper-elasticity
PK1_solid =  ufl.diff(Psi_solid, F)
PK1_medium =  ufl.diff(Psi_medium, F)

# create a scaler function to store face ID
# faceID = fem.Function(fem.functionspace(msh, element_disp))
# faceID.interpolate(lambda x: cell_tags.values[cell_tags.indices])
faceID = fem.Function(fem.functionspace(msh, element_lower))
faceID.x.array[cell_tags.indices] = cell_tags.values.astype(default_scalar_type)
print(f"faceID min: {faceID.x.array.min()}, max: {faceID.x.array.max()}")

# select the PK1 stress based on the face ID
PK1 = ufl.conditional(
    ufl.eq(faceID, VolID_solid), PK1_solid,
    ufl.conditional(ufl.eq(faceID, VolID_medium), PK1_medium, 0.0 * I)
)

# Cauchy stress tensor 
sigma = (1 / J) * (PK1 * F.T)
# Mises stress: sqrt(3/2 * s:s), where s is the deviatoric stress
sigma_s = sigma - (1 / 3) * tr(sigma) * I  # Deviatoric stress tensor
von_mises = ufl.sqrt(3.0 / 2.0 * ufl.inner(sigma_s, sigma_s))

# apply the boundary conditions =======================================

# To apply the clamp conditions, we locate the mesh facets that lie on the boundary
# We now find the degrees-of-freedom that are associated with the boundary facets using locate_dofs_topological:

# Define points
def pointA(x):
    return np.logical_and(np.isclose(x[0], 0.0, atol=1e-5), np.isclose(x[1], 0.0, atol=1e-5))

def pointB(x):
    return np.logical_and(np.isclose(x[0], 2.0, atol=1e-5), np.isclose(x[1], 0.0, atol=1e-5))

def pointC(x):
    return np.logical_and(np.isclose(x[0], 1.0, atol=1e-5), np.isclose(x[1], 0.5, atol=1e-5))

point_A = mesh.locate_entities(msh, fdim-1, pointA)
point_B = mesh.locate_entities(msh, fdim-1, pointB)
point_C = mesh.locate_entities(msh, fdim-1, pointC)

print(f"point_A: {point_A}")
print(f"point_B: {point_B}")
print(f"point_C: {point_C}")
# point_A: [0]
# point_B: [395]
# point_C: [284]

# Extract and collapse subspaces for each component
V_x, SubMap_x = V.sub(0).collapse()  # Subspace for u_x
V_y, SubMap_y = V.sub(1).collapse()  # Subspace for u_y
point_A_dofs = fem.locate_dofs_geometrical(V, pointA)
point_B_dofs_collapsed = fem.locate_dofs_geometrical((V.sub(1), V_y), pointB)
point_C_dofs_collapsed = fem.locate_dofs_geometrical((V.sub(1), V_y), pointC)
print(f"point_B_dofs_y_collapsed: {point_B_dofs_collapsed}")
print(f"point_C_dofs_y_collapsed: {point_C_dofs_collapsed}")
# point_B_dofs_y_collapsed: [array([791], dtype=int32), array([395], dtype=int32)]
# point_C_dofs_y_collapsed: [array([569], dtype=int32), array([284], dtype=int32)]
point_B_dofs_y = point_B_dofs_collapsed[0]
point_C_dofs_y = point_C_dofs_collapsed[0]
print(f"point_A_dofs: {point_A_dofs}")
print(f"point_B_dofs_y: {point_B_dofs_y}")
print(f"point_C_dofs_y: {point_C_dofs_y}")
# point_A_dofs: [0]
# point_B_dofs_y: [791]
# point_C_dofs_y: [569]

# use dirichletbc to create the boundary condition:
u_zero = np.array([0,0], dtype=default_scalar_type)

pointA_bc = fem.dirichletbc(u_zero, point_A_dofs, V)
# pointB_bc = fem.dirichletbc(u_zero, point_B_dofs, V)
pointB_bc = fem.dirichletbc(value=PETSc.ScalarType(0.0), dofs=point_B_dofs_y, V=V.sub(1))

u_Compress = fem.Constant(msh, 0.0)
pointC_bc = fem.dirichletbc(value=u_Compress, dofs=point_C_dofs_y, V=V.sub(1))

# assembly boundary conditions
bc_total = [pointA_bc, pointB_bc, pointC_bc]


# define the variational problem ======================================

# Define the volume integration measure "dx", also specify the quadrature degree for the integrals to 4.
dx = ufl.Measure('dx', domain=msh, subdomain_data=cell_tags, metadata={'quadrature_degree': 4})
ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags, metadata={'quadrature_degree': 4})

# Define residual force (we want to find u such that Res(u) = 0)
Res_solid = inner(PK1_solid, grad(u_test)) * dx(VolID_solid)
Res_medium = (inner(PK1_medium, grad(u_test))) * dx(VolID_medium) #  
# total Res
Res = Res_solid + Res_medium


dsol = TrialFunction(V)
dRes = derivative(Res, u, dsol)
# dRes is the Jacobian matrix for Newton iteration acclerating convergence. An explicit dRes is good for efficiency. 

# As the varitional form is non-linear and written on residual form, we use the non-linear problem class from DOLFINx to set up required structures to use a Newton solver.

problem = NonlinearProblem(Res, u, bc_total, dRes)

# create and customize the Newton solver  =============================
solver = NewtonSolver(msh.comm, problem)

# set the solver parameters
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 100
solver.report = True

#  The Krylov solver parameters.
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_max_it"] = 30
ksp.setFromOptions()

# output solution for visualization ===================================

# Create VTK writer for output
from pathlib import Path
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)

# displacement output
VTKfile_result = io.VTKFile(msh.comm, str(results_folder / "Box.pvd"), "w")
VTKfile_result.write_mesh(msh)
u_proj = fem.Function(fem.functionspace(msh, element_disp))
u_proj.name = "displacement" # set the name 

# p output
# p_proj = fem.Function(fem.functionspace(msh, element_p))
# p_proj.name = "p" # set the name 

# q output
# q_proj = fem.Function(fem.functionspace(msh, element_q))
# q_proj.name = "q" # set the name 

# define function space for von_mises stress
V_von_mises = fem.functionspace(msh, element_lower)
stress_expr = fem.Expression(von_mises, V_von_mises.element.interpolation_points())
stresses = fem.Function(V_von_mises)
stresses.name = "von-Mises stress"  # set the name

# function to write results to VTK file at time t
def writeResults(sol, stresses, time_cur):
    # interpolation
    u_proj.interpolate(sol)
    # p_proj.interpolate(sol.sub(1))
    # q_proj.interpolate(sol.sub(2))
    # calculate von Mises stress
    stresses.interpolate(stress_expr)
    
    # write file
    VTKfile_result.write_function([u_proj, stresses], time_cur)

# simulation start ====================================================
print("-----------------------------------")
print("Simulation has started")
print("-----------------------------------")

# Simulation parameters
num_steps = 100
time_final = 1.0
dt = time_final/num_steps

# Initial time and step
timeStep = 0
time_cur = 0.0

# Loop over time steps
for timeStep in range(1, num_steps + 1):
    # Update current time
    time_cur = timeStep * dt

    # TODO: Update the material parameters = initial + (target - initial)*time_cur
    #Gamma.value = 2.0e-2 + (2.0e-3 - 2.0e-2)*time_cur
    #alpha_r.value = 0.1 + (0.5 - 0.1)*time_cur

    # the simulations runs for these parameters
    #Gamma.value = 0.01
    #alpha_r.value = 0.1

    Gamma.value = 0.01
    alpha_r.value = 10.0

    # Update applied displacement
    u_Compress.value = -1*time_cur
    print(f"at time = {time_cur}, uy_Compress={u_Compress.value}")

    # Solve the problem
    num_its, converged = solver.solve(u)
    if converged:
        print(f"Time step {timeStep}: Converged in {num_its} iterations.")
    else:
        print(f"Time step {timeStep}: Not converged.")
        print(f"Residual norm: {solver.r_norm}")

    # Write results for current time step
    writeResults(u, stresses, time_cur)

# Close the VTK files
VTKfile_result.close()

print("-----------------------------------")
print("Simulation completed successfully.")
print("-----------------------------------")
