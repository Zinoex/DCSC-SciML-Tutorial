# Credit: SciML Automatically Discover Missing Physics by Embedding Machine Learning into Differential Equations
# Link: https://docs.sciml.ai/Overview/stable/showcase/missing_physics/


# With real data, check: https://github.com/ChrisRackauckas/universal_differential_equations/blob/master/LotkaVolterra/hudson_bay.jl


# SciML Tools
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

# Standard Libraries
using LinearAlgebra, Statistics

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, StableRNGs, Measurements
gr()

# Set a random seed for reproducible behaviour
rng = StableRNG(1111)






########################################################
# Generate training data for the Lotka-Volterra system #
########################################################
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Define the experimental parameter
tspan = (0.0, 5.0) 
u0 = 5.0f0 * rand(rng, 2)  # Initial conditions for the Lotka-Volterra system (x₀, y₀) ∈ [0, 5]²
p_ = [1.3, 0.9, 0.8, 1.8]  # α, β, γ, δ

prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

# Add noise in terms of the mean
X = Array(solution)
t = solution.t

x̄ = mean(X, dims = 2)
noise_magnitude = 5e-3
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

# Plot the true and noisy data
plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])






###################################
# Universal Differential Equation #
###################################
rbf(x) = exp.(-(x .^ 2))

# Multilayer feed-forward network
const U = Lux.Chain(
    Lux.Dense(2, 5, rbf),
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2)
)

# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)
const _st = st

# Define the hybrid model
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p, _st)[1] # Network prediction - NN(x, θ) = û
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)

# Define the problem
prob_nn = ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, p)





#######################
# Setup loss function #
#######################
function predict(θ, X = Xₙ[:, 1], T = t)
    # Reconstruct the problem with the new parameters θ
    # and potentially also new initial conditions X and time span T
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)

    # Solve the IVP
    solution = solve(_prob, Vern7(), saveat = T, abstol = 1e-6, reltol = 1e-6,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)))

    return Array(solution)
end

function loss(θ)  # This could also take trajectory data as an argument
    X̂ = predict(θ)

    return mean(abs2, Xₙ .- X̂) # Mean squared error
end


losses = Float64[]
function callback(state, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end




############
# Training #
############

# The optimization function
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)

# First use ADAM to get close to the minimum
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(), callback = callback, maxiters = 5000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

# Then use L-BFGS-B to refine the solution
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, LBFGS(linesearch = BackTracking()), callback = callback, maxiters = 1000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Rename the best candidate
p_trained = res2.u






################
# Plot Results #
################

## Plot the losses
pl_losses = plot(1:5000, losses[1:5000], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(5001:length(losses), losses[5001:end], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "LBFGS", color = :red)


## Analysis of the trained network
# Plot the data and the approximation
ts = first(solution.t):(mean(diff(solution.t)) / 2):last(solution.t)
X̂ = predict(p_trained, Xₙ[:, 1], ts)
# Trained on noisy data vs real solution
pl_trajectory = plot(ts, transpose(X̂), xlabel = "t", ylabel = "x(t), y(t)", color = :red, label = ["UDE Approximation" nothing])
scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Measurements" nothing])



## Ideal unknown interactions of the predictor
Ȳ = [-p_[2] * (X̂[1, :] .* X̂[2, :])'; p_[3] * (X̂[1, :] .* X̂[2, :])']
# Neural network guess
Ŷ = U(X̂, p_trained, st)[1]

pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel = "U(x,y)", color = :red, label = ["UDE Approximation" nothing])
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])



## Plot the error
pl_reconstruction_error = plot(ts, norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, xlabel = "t",
    ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2, 1))
pl_overall = plot(pl_trajectory, pl_missing)






###############################################
# Symbolic regression (via sparse regression) #
###############################################

@variables u[1:2]
b = polynomial_basis(u, 4)
basis = Basis(b, u);

λ = 1e-1
opt = ADMM(λ)

## Find argmin ‖̇X - B(X)Ξ‖₂ + λ‖Ξ‖₁ - do not assume any dynamics
full_problem = ContinuousDataDrivenProblem(Xₙ, t)

options = DataDrivenCommonOptions(maxiters = 10_000,
    normalize = DataNormalization(ZScoreTransform),
    selector = bic, digits = 1,
    data_processing = DataProcessing(split = 0.9,
        batchsize = 30,
        shuffle = true,
        rng = StableRNG(1111)))

full_res = solve(full_problem, basis, opt, options = options)
full_eqs = get_basis(full_res)
println(full_res)

## Find argmin ‖Y - B(X)Ξ‖₂ + λ‖Ξ‖₁ - only interpolate the missing physics
nn_problem = DirectDataDrivenProblem(X̂, Ŷ)

options = DataDrivenCommonOptions(maxiters = 10_000,
    normalize = DataNormalization(ZScoreTransform),
    selector = bic, digits = 1,
    data_processing = DataProcessing(split = 0.9,
        batchsize = 30,
        shuffle = true,
        rng = StableRNG(1111)))

nn_res = solve(nn_problem, basis, opt, options = options)
nn_eqs = get_basis(nn_res)
println(nn_res)


## Analyze the results
for eqs in (full_eqs, nn_eqs)
    println(eqs)
    println(get_parameter_map(eqs))
    println()
end


## Define the recovered, hybrid model
function recovered_dynamics!(du, u, p, t)
    û = nn_eqs(u, p) # Recovered equations
    du[1] = p_[1] * u[1] + û[1]
    du[2] = -p_[4] * u[2] + û[2]
end

estimation_prob = ODEProblem(recovered_dynamics!, u0, tspan, get_parameter_values(nn_eqs))
estimate = solve(estimation_prob, Tsit5(), saveat = solution.t)

# Plot
plot(solution, label = ["True u[1]" "True u[2]"])
plot!(estimate, label = ["Recovered u[1]" "Recovered u[2]"])