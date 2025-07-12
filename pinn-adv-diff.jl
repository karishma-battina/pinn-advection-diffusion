using NeuralPDE, Lux, Optimization, OptimizationOptimisers, Plots
using LinearAlgebra
using ModelingToolkit: Interval
using Random, QuasiMonteCarlo, Printf, Statistics, ProgressMeter, BenchmarkTools

# Parameters
const T_final = 0.25
const X_max = 1.0
const Y_max = 1.0
const d = 0.01  # Diffusion coefficient
const vx, vy = 0.5, 0.5  # Advection velocities

# Noise Params
const noise_level_bc_abs = 0.01
const noise_level_ic_rel = 0.005
const noise_level_data_rel = 0.005  # Noise for FDM data

# Loss Function Weighting
const data_loss_weight = 10.0  # Weight for FDM data fitting term
const ic_loss_weight = 500.0   # Weight for IC fitting term
const data_batch_size = 1024   # Batch size for FDM data
const ic_batch_size = 1024     # Batch size for IC data

# Initial condition function
target_ic(x, y) = exp(-100 * ((x - 0.5)^2 + (y - 0.5)^2))


Random.seed!(1234)

# Generate FDM data parameters
Nx, Ny, Nt = 51, 51, 100
dx, dy = X_max/(Nx-1), Y_max/(Ny-1)
dt = T_final/Nt
x_fdm = range(0.0, stop = X_max, length = Nx)
y_fdm = range(0.0, stop = Y_max, length = Ny)
t_fdm = range(0.0, stop = T_final, length = Nt+1)
U = zeros(Float64, Nx, Ny, Nt+1)

# Initialize IC and BC
for i in 1:Nx, j in 1:Ny
    U[i,j,1] = target_ic(x_fdm[i], y_fdm[j])
end
U[1,:,:] .= 0.0; U[Nx,:,:] .= 0.0; U[:,1,:] .= 0.0; U[:,Ny,:] .= 0.0

# Run and time FDM solver
println("Running and timing FDM solver...")
time_fdm = @belapsed begin
    for n in 1:Nt
        Un = @view U[:,:,n]
        Un1 = @view U[:,:,n+1]
        for i in 2:Nx-1, j in 2:Ny-1
            adv = vx*(Un[i,j] - Un[i-1,j])/dx + vy*(Un[i,j] - Un[i,j-1])/dy
            diff = d*((Un[i+1,j] - 2*Un[i,j] + Un[i-1,j])/dx^2 +
                      (Un[i,j+1] - 2*Un[i,j] + Un[i,j-1])/dy^2)
            Un1[i,j] = Un[i,j] - dt*adv + dt*diff
        end
        Un1[1,:] .= 0.0; Un1[Nx,:] .= 0.0; Un1[:,1] .= 0.0; Un1[:,Ny] .= 0.0
    end
end
println("FDM solver time: ", time_fdm)
println("Finished FDM solver. Inspecting U[:,:,end] at T_final:")
println("  Max: ", maximum(abs.(U[:,:,end])))
println("  Min: ", minimum(U[:,:,end]))
println("  Norm: ", norm(vec(U[:,:,end])))

# Add noise and flatten data
U_noisy = U .+ noise_level_data_rel * std(U) .* randn(size(U))
num_pts = length(U_noisy)
t_data, x_data, y_data, u_data = Float64[], Float64[], Float64[], Float64[]
for k in 1:Nt+1, j in 1:Ny, i in 1:Nx
    push!(t_data, t_fdm[k]); push!(x_data, x_fdm[i]); push!(y_data, y_fdm[j])
    push!(u_data, U_noisy[i,j,k])
end

# Define PINN problem
@parameters t,x,y
@variables u(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
Dxx, Dyy = Dx^2, Dy^2
pde = Dt(u(t,x,y)) + vx*Dx(u(t,x,y)) + vy*Dy(u(t,x,y)) ~
      d*(Dxx(u(t,x,y)) + Dyy(u(t,x,y)))
noisy_b(t,x,y) = noise_level_bc_abs * randn()
bcs = [
    u(t,x,0) ~ noisy_b(t,x,0),
    u(t,x,1) ~ noisy_b(t,x,1),
    u(t,0,y) ~ noisy_b(t,0,y),
    u(t,1,y) ~ noisy_b(t,1,y),
    u(0,x,y) ~ target_ic(x,y)
]
doms = [x ∈ Interval(0,X_max), y ∈ Interval(0,Y_max), t ∈ Interval(0,T_final)]
chain = Lux.Chain(
    Lux.Dense(3,256,Lux.tanh),
    Lux.Dense(256,256,Lux.tanh), Lux.Dense(256,256,Lux.tanh), Lux.Dense(256,256,Lux.tanh),
    Lux.Dense(256,256,Lux.tanh), Lux.Dense(256,256,Lux.tanh), Lux.Dense(256,256,Lux.tanh),
    Lux.Dense(256,256,Lux.tanh), Lux.Dense(256,1)
)

# Loss definitions
function data_loss(phi, p, _)
    idx = rand(1:num_pts, data_batch_size)
    pts = Float32.([t_data[idx]'; x_data[idx]'; y_data[idx]'])
    pred = vec(phi(pts, p)); ref = Float32.(u_data[idx])
    return data_loss_weight * sum((pred .- ref).^2)
end
function ic_loss(phi, p, _)
    x_ic = X_max * rand(ic_batch_size)
    y_ic = Y_max * rand(ic_batch_size)
    pts = Float32.([zeros(ic_batch_size)'; x_ic'; y_ic'])
    pred = vec(phi(pts, p)); ref = Float32.(target_ic.(x_ic, y_ic))
    return ic_loss_weight * sum((pred .- ref).^2)
end
loss_comb(phi, p, extra) = data_loss(phi,p,extra) + ic_loss(phi,p,extra)

# Discretize the PDE
disc = PhysicsInformedNN(chain, QuasiRandomTraining(200;
           sampling_alg=QuasiMonteCarlo.LatinHypercubeSample()),
           additional_loss=loss_comb)
@named sys = PDESystem(pde, bcs, doms, [t,x,y], [u(t,x,y)])
prob = discretize(sys, disc)
if eltype(prob.u0) == Float64
    prob = remake(prob, u0=Float32.(prob.u0))
end

# Train with Adam
adam_hist = Float32[]; it_adam = Ref(0)
cb_adam = (p,l) -> begin
    it_adam[] += 1
    push!(adam_hist, l)
    if it_adam[] % 100 == 0
        @printf("Adam %d: %.4e\n", it_adam[], l)
    end
    false
end
opt_adam = OptimizationOptimisers.Adam(0.006)
time_ad = @belapsed(global res_ad = Optimization.solve(prob, $opt_adam, callback=cb_adam, maxiters=330))


@printf("Final loss after Adam: %.4e\n", res_ad.objective)
println("Adam time: ", time_ad)

# Compute relative L2 error at final time using Adam solution
eu_end = U[:,:,end]
phi_fun = disc.phi
u_pred = [first(phi_fun(Float32.([T_final; x; y]), res_ad.u)) for x in x_fdm, y in y_fdm]
rel_error = norm(u_pred .- eu_end) / norm(eu_end)
println("Relative L2 error at t_final: ", rel_error)

# Visualization of solutions and Adam loss history
xs_vis, ys_vis = x_fdm, y_fdm
u_ic_grid = [target_ic(x,y) for x in xs_vis, y in ys_vis]

p1 = heatmap(xs_vis, ys_vis, u_ic_grid', title="Clean IC (t=0)", xlabel="x", ylabel="y")
u_pinn_ic = [first(phi_fun(Float32.([0.0; x; y]), res_ad.u)) for x in xs_vis, y in ys_vis]
p2 = heatmap(xs_vis, ys_vis, u_pinn_ic', title="PINN IC (t=0)")
p3 = heatmap(xs_vis, ys_vis, eu_end', title="FDM (t=T_final)")
p4 = heatmap(xs_vis, ys_vis, u_pred', title="PINN (t=T_final)")

p_heat = plot(p1, p2, p3, p4, layout=(2,2), size=(1300,900), dpi=100)

# Adam loss history plot
p_loss_adam = plot(1:length(adam_hist), adam_hist, label="Adam Loss", xlabel="Iteration", ylabel="Loss", title="Adam Loss History", yscale=:log10)

# Display results
display(p_heat)
display(p_loss_adam)
