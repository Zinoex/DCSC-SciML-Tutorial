# Credit: SciML Optimization Under Uncertainty
# Link: https://docs.sciml.ai/Overview/stable/showcase/optimization_under_uncertainty/

# Imports
using DifferentialEquations, Plots
using Distributions, SciMLExpectations
using Optimization, OptimizationNLopt, OptimizationMOI


#############################
# Dynamics u = (x, ̇x, y, ̇y) #
#############################
function ball!(du, u, p, t)
    du[1] = u[2]
    du[2] = 0.0
    du[3] = u[4]
    du[4] = -p[1]
end

ground_condition(u, t, integrator) = u[3]  # Detect when the ball hits the ground
function ground_affect!(integrator)
    integrator.u[4] = -integrator.p[2] * integrator.u[4]  # Reverse the vertical velocity

    return false
end
ground_cb = ContinuousCallback(ground_condition, ground_affect!)

u0 = [0.0, 2.0, 50.0, 0.0]  # Zero initial vertical velocity
tspan = (0.0, 50.0)
p = [9.807, 0.9]  # g, coefficient of restitution

prob = ODEProblem(ball!, u0, tspan, p)
sol = solve(prob, Tsit5(), callback = ground_cb)

# Plot the trajectory
plot(sol, vars = (1, 3), label = nothing, xlabel = "x", ylabel = "y")




######################################
# Stop when hitting a wall at x = 25 #
######################################
stop_condition(u, t, integrator) = u[1] - 25.0
stop_cb = ContinuousCallback(stop_condition, terminate!)
cbs = CallbackSet(ground_cb, stop_cb)

tspan = (0.0, 1500.0)

prob = ODEProblem(ball!, u0, tspan, p)
sol = solve(prob, Tsit5(), callback = cbs)

# Plot the trajectory
rectangle(xc, yc, w, h) = Shape(xc .+ [-w, w, w, -w] ./ 2.0, yc .+ [-h, -h, h, h] ./ 2.0)

begin
    plot(sol, vars = (1, 3), label = nothing, lw = 3, c = :black)  # Plot the trajectory
    xlabel!("x [m]")
    ylabel!("y [m]")
    plot!(rectangle(27.5, 25, 5, 50), c = :red, label = nothing)    # Plot the wall
    scatter!([25], [25], marker = :star, ms = 10, label = nothing, c = :green)   # Plot the target
    ylims!(0.0, 50.0)
end





######################
# Adding uncertainty #
######################

cor_dist = truncated(Normal(0.9, 0.02), 0.9 - 3 * 0.02, 1.0)
trajectories = 100

prob_func(prob, i, repeat) = remake(prob, p = [p[1], rand(cor_dist)])
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
ensemblesol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = trajectories, callback = cbs)

begin # plot
    plot(ensemblesol, vars = (1, 3), lw = 1)  # Plot trajectories for samples of the CoR
    xlabel!("x [m]")
    ylabel!("y [m]")
    plot!(rectangle(27.5, 25, 5, 50), c = :red, label = nothing)   # Plot the wall
    scatter!([25], [25], marker = :star, ms = 10, label = nothing, c = :green)  # Plot the target
    plot!(sol, vars = (1, 3), label = nothing, lw = 3, c = :black, ls = :dash)  # Plot the deterministic trajectory
    xlims!(0.0, 27.5)
end


# Goal as observable
obs(sol, p) = abs2(sol[3, end] - 25)

mean_ensemble = mean([obs(sol, p) for sol in ensemblesol])
println(mean_ensemble)

gd = GenericDistribution(cor_dist)
h(x, u, p) = u, [p[1]; x[1]]
sm = SystemMap(prob, Tsit5(), callback = cbs)
exprob = ExpectationProblem(sm, obs, h, gd)
sol = solve(exprob, Koopman(), ireltol = 1e-5)
println(sol.u)


####################################
# Optimizing the initial condition #
####################################

# x₀ ∈ [-100, 0], ̇x₀ ∈ [1, 3], y₀ ∈ [10, 50]

make_u0(θ) = [θ[1], θ[2], θ[3], 0.0]  # Zero initial vertical velocity
function 𝔼_loss(θ, pars)
    prob = ODEProblem(ball!, make_u0(θ), tspan, p)
    sm = SystemMap(prob, Tsit5(), callback = cbs)
    exprob = ExpectationProblem(sm, obs, h, gd)
    sol = solve(exprob, Koopman(), ireltol = 1e-5)
    sol.u
end
opt_f = OptimizationFunction(𝔼_loss, Optimization.AutoForwardDiff())

# Optimization bounds and initial condition
opt_ini = [-1.0, 2.0, 50.0]
opt_lb = [-100.0, 1.0, 10.0]
opt_ub = [0.0, 3.0, 50.0]
opt_prob = OptimizationProblem(opt_f, opt_ini; lb = opt_lb, ub = opt_ub)

# Optimize
optimizer = OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :LD_MMA)
opt_sol = solve(opt_prob, optimizer)

minx = opt_sol.u

# Plot the optimized trajectories
ensembleprob = EnsembleProblem(remake(prob, u0 = make_u0(minx)), prob_func = prob_func)
ensemblesol = solve(ensembleprob, Tsit5(), EnsembleThreads(), trajectories = 100, callback = cbs)

begin
    plot(ensemblesol, vars = (1, 3), lw = 1, alpha = 0.1)    # Plot trajectories for samples of the CoR
    plot!(solve(remake(prob, u0 = make_u0(minx)), Tsit5(), callback = cbs),
        vars = (1, 3), label = nothing, c = :black, lw = 3, ls = :dash)  # Plot the optimized trajectory
    xlabel!("x [m]")
    ylabel!("y [m]")
    plot!(rectangle(27.5, 25, 5, 50), c = :red, label = nothing)  # Plot the wall
    scatter!([25], [25], marker = :star, ms = 10, label = nothing, c = :green)  # Plot the target
    ylims!(0.0, 50.0)
    xlims!(minx[1], 27.5)
end



# We can also add chance constraints to the optimization problem (see the showcase for more details)