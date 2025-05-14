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
degree_disp = 1
degree_lower = degree_disp - 1 

# Create element, TODO: which degree should be used for p and q? they are gradients of displacement.
element_disp = element("Lagrange", msh.topology.cell_name(), degree_disp, shape=(gdim,))
element_p = element("Lagrange", msh.topology.cell_name(), degree_disp) 
element_q = element("Lagrange", msh.topology.cell_name(), degree_disp) 
# element_p = element("DG", msh.topology.cell_name(), degree_lower)
# element_q = element("DG", msh.topology.cell_name(), degree_lower)
element_lower = element("DG", msh.topology.cell_name(), degree_lower)

# create a function space V on the mesh
V_ME = fem.functionspace(msh, mixed_element([element_disp, element_p, element_q]))

# defining the trial(unknown) and test(small variation) function
sol = fem.Function(V_ME) # for non-linear problem
u, p, q = ufl.split(sol) # returns the components of the mixed function

u_test, p_test, q_test = TestFunctions(V_ME) # test functions for mixed spaces can be split

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
phi = ufl.variable((F[0,1]-F[1,0])/(F[0,0]+F[1,1] + eps_))

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
Gamma = fem.Constant(msh, PETSc.ScalarType(2.0e-1)) # gamma = 2.0e-4, from Section 4.1
# the penalty parameters which enforce the approximation of the gradient
beta1 = fem.Constant(msh, PETSc.ScalarType(1e+4)) # taken from Section 4.2.1
beta2 = fem.Constant(msh, PETSc.ScalarType(1e+4)) # taken from Section 4.2.1
# the penalty parameter which enforces the regularization
alpha_r = fem.Constant(msh, PETSc.ScalarType(0.1)) # alpha_r=100 from Section 4.1
# strain energy density (compressible/incompressible neo-Hookean model)
# Psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
# Psi  = 0.5*mu_active*(ICbar-3) + p*(J-1)

# elastomer solid in Wriggers's CMAME paper 2025
Psi_solid = (bulkmod/2) * (ufl.ln(J))**2 + (shearmod/2) * (J**(-2.0/3.0) * Ic - 3)
# third medium, in the case of 2D. 
# This formulation approximates \phi and J with p and q, respectively, and includes penalty terms to enforce these approximations.
# p and q are continuous fields and their contributions are defined only in the third medium.
# the regularization term \nabla J can be omitted when using finite elements with linear shape functions.
W_Phi = 0.5*Gamma*(beta1 * (phi - p)**2 + alpha_r * inner(ufl.grad(p), ufl.grad(p)))
W_J = 0.5*Gamma*(beta2 * (J - q)**2 + alpha_r * inner(ufl.grad(q), ufl.grad(q)))
Psi_medium = Gamma*(shearmod/2) * (J**(-2.0/3.0) * Ic - 3) + W_J + W_Phi  # keep consistent with paper, by omitting the first term in Psi_solid

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

# Collapse the displacement subspace
V_disp, SubMap = V_ME.sub(0).collapse()
# Extract and collapse subspaces for each component
V_disp_x, SubMap_x = V_ME.sub(0).sub(0).collapse()  # Subspace for u_x
V_disp_y, SubMap_y = V_ME.sub(0).sub(1).collapse()  # Subspace for u_y
# Locate DOFs geometrically for displacement
dofs_point_A_collapsed = fem.locate_dofs_geometrical((V_ME.sub(0),V_disp), pointA)
dofs_point_B_collapsed = fem.locate_dofs_geometrical((V_ME.sub(0),V_disp), pointB)
dofs_point_C_collapsed = fem.locate_dofs_geometrical((V_ME.sub(0),V_disp), pointC)
print(f"dofs_point_A_collapsed: {dofs_point_A_collapsed}")
print(f"dofs_point_B_collapsed: {dofs_point_B_collapsed}")
print(f"dofs_point_C_collapsed: {dofs_point_C_collapsed}")
# [array([0, 1], dtype=int32), array([0, 1], dtype=int32)]
# [array([1580, 1581], dtype=int32), array([880, 881], dtype=int32)]
# [array([1136, 1137], dtype=int32), array([460, 461], dtype=int32)]

# Locate DOFs geometrically for each component
dofs_point_A_ux_collapsed = fem.locate_dofs_geometrical((V_ME.sub(0).sub(0),V_disp_x), pointA)
dofs_point_A_uy_collapsed = fem.locate_dofs_geometrical((V_ME.sub(0).sub(1),V_disp_y), pointA)
dofs_point_B_uy_collapsed = fem.locate_dofs_geometrical((V_ME.sub(0).sub(1),V_disp_y), pointB)  
dofs_point_C_uy_collapsed = fem.locate_dofs_geometrical((V_ME.sub(0).sub(1),V_disp_y), pointC)  
print(f"dofs_point_A_ux_collapsed: {dofs_point_A_ux_collapsed}")
print(f"dofs_point_A_uy_collapsed: {dofs_point_A_uy_collapsed}")
print(f"dofs_point_B_uy_collapsed: {dofs_point_B_uy_collapsed}")
print(f"dofs_point_C_uy_collapsed: {dofs_point_C_uy_collapsed}")
# [array([0], dtype=int32), array([0], dtype=int32)]
# [array([1], dtype=int32), array([0], dtype=int32)]
# [array([1581], dtype=int32), array([440], dtype=int32)]
# [array([1137], dtype=int32), array([230], dtype=int32)]

dofs_point_A_ux = dofs_point_A_ux_collapsed[0]
dofs_point_A_uy = dofs_point_A_uy_collapsed[0]
dofs_point_B_uy = dofs_point_B_uy_collapsed[0]
dofs_point_C_uy = dofs_point_C_uy_collapsed[0]
print(f"dofs_point_A_ux: {dofs_point_A_ux}")
print(f"dofs_point_A_uy: {dofs_point_A_uy}")
print(f"dofs_point_B_uy: {dofs_point_B_uy}")
print(f"dofs_point_C_uy: {dofs_point_C_uy}")
# dofs_point_A_ux: [0]
# dofs_point_A_uy: [1]
# dofs_point_B_uy: [1581]
# dofs_point_C_uy: [1137]

# use dirichletbc to create the boundary condition:
# according to Section 4.1.1, the box is fixed at the lower left corner in all directions and simply supported at the lower right corner.
zero = np.array([0, 0], dtype=default_scalar_type)
u_Compress = fem.Constant(msh, 0.0)

bc_point_A_ux = fem.dirichletbc(value=PETSc.ScalarType(0.0), dofs=dofs_point_A_ux, V=V_ME.sub(0).sub(0))
bc_point_A_uy = fem.dirichletbc(value=PETSc.ScalarType(0.0), dofs=dofs_point_A_uy, V=V_ME.sub(0).sub(1))
bc_point_B_uy = fem.dirichletbc(value=PETSc.ScalarType(0.0), dofs=dofs_point_B_uy, V=V_ME.sub(0).sub(1))
bc_point_C_uy = fem.dirichletbc(value=u_Compress, dofs=dofs_point_C_uy, V=V_ME.sub(0).sub(1))

# assembly boundary conditions
bc_total = [bc_point_A_ux, bc_point_A_uy, bc_point_B_uy, bc_point_C_uy]
# TODO: how to check if the boundary conditions are applied correctly, how to visualize them?

# define the variational problem ======================================

# Define the volume integration measure "dx", also specify the quadrature degree for the integrals to 4.
dx = ufl.Measure('dx', domain=msh, subdomain_data=cell_tags, metadata={'quadrature_degree': 4})
ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags, metadata={'quadrature_degree': 4})

# Define residual force (we want to find u such that Res(u) = 0)
Res_solid = inner(PK1_solid, grad(u_test)) * dx(VolID_solid)
Res_medium = (inner(PK1_medium, grad(u_test)) +
              Gamma * (beta1 * inner(p - phi, p_test) + alpha_r * inner(grad(p), grad(p_test) )) +
              Gamma * (beta2 * inner(q - J, q_test) + alpha_r * inner(grad(q), grad(q_test) )) ) * dx(VolID_medium) 
# total Res
Res = Res_solid + Res_medium #- dot(traction, u_test)*ds(SurID_right)

# (3) is a facet tag used to indicate a specific boundary or surface portion, such as the physical surface labeled ID 3 in the Gmsh file.

dsol = TrialFunction(V_ME)
dRes = derivative(Res, sol, dsol)
# dRes is the Jacobian matrix for Newton iteration acclerating convergence. An explicit dRes is good for efficiency. 

# As the varitional form is non-linear and written on residual form, we use the non-linear problem class from DOLFINx to set up required structures to use a Newton solver.

problem = NonlinearProblem(Res, sol, bc_total, dRes)

# create and customize the Newton solver  =============================
solver = NewtonSolver(msh.comm, problem)

# set the solver parameters
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
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
results_folder = Path("results2")
results_folder.mkdir(exist_ok=True, parents=True)

# displacement output
VTKfile_result = io.VTKFile(msh.comm, str(results_folder / "Box.pvd"), "w")
VTKfile_result.write_mesh(msh)
u_proj = fem.Function(fem.functionspace(msh, element_disp))
u_proj.name = "displacement" # set the name 

# p output
p_proj = fem.Function(fem.functionspace(msh, element_p))
p_proj.name = "p" # set the name 

# q output
q_proj = fem.Function(fem.functionspace(msh, element_q))
q_proj.name = "q" # set the name 

# define function space for von_mises stress
V_von_mises = fem.functionspace(msh, element_lower)
stress_expr = fem.Expression(von_mises, V_von_mises.element.interpolation_points())
stresses = fem.Function(V_von_mises)
stresses.name = "von-Mises stress"  # set the name

# function to write results to VTK file at time t
def writeResults(sol, stresses, time_cur):
    # interpolation
    u_proj.interpolate(sol.sub(0))
    p_proj.interpolate(sol.sub(1))
    q_proj.interpolate(sol.sub(2))
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

    # Update traction
    traction.value = (traction_x * time_cur, traction_y * time_cur, traction_z * time_cur)

    # TODO: Update the material parameters = initial + (target - initial)*time_cur
    Gamma.value = 2.0e-1 + (2.0e-2 - 2.0e-1)*time_cur
    alpha_r.value = 0.01 + (0.5 - 0.01)*time_cur

    # Update applied displacement
    u_Compress.value = -0.000000*time_cur
    print(f"at time = {time_cur}, uy_Compress={u_Compress.value}")

    # Solve the problem
    num_its, converged = solver.solve(sol)
    if converged:
        print(f"Time step {timeStep}: Converged in {num_its} iterations.")
    else:
        print(f"Time step {timeStep}: Not converged.")
        print(f"Residual norm: {solver.r_norm}")

    # Write results for current time step
    writeResults(sol, stresses, time_cur)

# Close the VTK files
VTKfile_result.close()

print("-----------------------------------")
print("Simulation completed successfully.")
print("-----------------------------------")
