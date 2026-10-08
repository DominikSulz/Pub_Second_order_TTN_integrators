# ============================================================================
# Single-file script: generate all convergence data and plot it.
#   - reference solution (stochastic collocation), computed or read from disk
#   - parallel Tucker integrator, order 1 and order 2
# Needs settings.jl, Rhs.jl, Problem.jl, TTN.jl, Solver.jl, solverDeterministic.jl
# in the same folder.
# ============================================================================
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Base.Threads
using ProgressMeter
include("settings.jl")
include("Rhs.jl")
include("Problem.jl")
include("Solver.jl")
include("solverDeterministic.jl")
using PyCall
using PyPlot
np = pyimport("numpy")
using DelimitedFiles
using Random

Random.seed!(1234)
close("all")

# ------------------------------- user options -------------------------------
const RERUN_COLLOCATION = false              # true: recompute reference (slow) and save it
const RERUN_TUCKER      = true               # false: reuse csv files from earlier runs
const EPS_VALUES        = [1e-3, 1e-4]       # truncation tolerances ϑ
const NAIVE             = false              # first-order approximation from the first-order paper
const NDT               = 10                 # number of time steps sizes
# ----------------------------------------------------------------------------

function computeMoments(ρ, n, wξ)
    Eρ = zeros(n)
    for j in 1:n
        Eρ[j] += sum(0.5 * wξ .* ρ[j, :])
    end
    σ² = zeros(n)
    for j in 1:n
        σ²[j] += sum(0.5 * wξ .* (ρ[j, :] .- Eρ[j]).^2)
    end
    return Eρ, σ²
end

function make_settings(ϵ)
    s = Settings()
    s.problem = "radiationUQ"
    s.ϵ = ϵ
    return s
end

# ------------------------------ reference solution --------------------------
s0 = make_settings(EPS_VALUES[1])
n  = s0.NCells
nξ = s0.Nxi
ξ, wξ = gausslegendre(nξ)

fρ, fE, fσ = "rhoCollocation1D_Nx$(n).txt", "rhoECollocation1D_Nx$(n).txt", "rhoSigCollocation1D_Nx$(n).txt"

if !RERUN_COLLOCATION && isfile(fρ) && isfile(fE) && isfile(fσ)
    ρCol  = readdlm(fρ, ',')
    EρCol = vec(readdlm(fE, ','))
    σ²Col = vec(readdlm(fσ, ','))
else
    println("Computing collocation reference solution ...")
    ρCol = zeros(n, nξ)
    @time Threads.@threads for k in eachindex(ξ)
        println("iteration ", k)
        scol_local = Settings()                 # own instance per thread
        scol_local.Δt = 0.01 * s0.Δt
        solver_local = SolverDeterministic(scol_local)
        ρCol[:, k] = Solve(solver_local, ξ[k])[:, 1]
    end
    ρCol  = ρCol[:, end:-1:1]
    EρCol = vec(sum(0.5 .* wξ' .* ρCol, dims=2))
    σ²Col = vec(sum(0.5 .* wξ' .* (ρCol .- EρCol).^2, dims=2))
    writedlm(fρ, ρCol, ',')
    writedlm(fE, EρCol, ',')
    writedlm(fσ, σ²Col, ',')
end

# ------------------------------ convergence runs ----------------------------
function convergence_run(ϵ, order; naive=false)
    s = make_settings(ϵ)
    Δt0 = s.Δt
    Δt_values = Δt0 * (10).^(-range(0, 1; length=NDT))

    err_ρ  = zeros(NDT)
    err_Eρ = zeros(NDT)
    err_σ² = zeros(NDT)
    ρ = zeros(n, nξ)

    for (i, Δt) in enumerate(Δt_values)
        s.Δt = Δt
        solver = Solver(s)
        if order == 1
            t, Y, rBUG = ParallelTuckerIntegrator(solver, true, naive)
        else
            t, Y, rBUG = ParallelTuckerIntegrator2ndOrder(solver)
        end

        ρ .= Y[:, 1, :]
        Eρ, σ² = computeMoments(ρ, s.NCells, wξ)

        err_ρ[i]  = sqrt(s.Δx * sum((ρ .- ρCol).^2))
        err_Eρ[i] = sqrt(s.Δx * sum((Eρ .- EρCol).^2))
        err_σ²[i] = sqrt(s.Δx * sum((σ² .- σ²Col).^2))
        println("order ", order, ", ϑ = ", ϵ, ", Δt = ", Δt,
                " err_ρ = ", err_ρ[i], " err_Eρ = ", err_Eρ[i], " err_σ² = ", err_σ²[i])
    end

    # save csv
    s.Δt = Δt0
    tag = order == 1 ? (naive ? "order1_naive_" : "order1_") : ""
    filename = "convergence_$(tag)$(s.problem)_Nx$(s.Nx)_nPN$(s.nPN)_Nxi$(s.Nxi)_Neta$(s.Neta)_tEnd$(s.tEnd)_eps$(s.ϵ).csv"
    open(filename, "w") do io
        println(io, "# problem=$(s.problem)")
        println(io, "# Nx=$(s.Nx)")
        println(io, "# nPN=$(s.nPN)")
        println(io, "# Nxi=$(s.Nxi)")
        println(io, "# Neta=$(s.Neta)")
        println(io, "# tEnd=$(s.tEnd)")
        println(io, "# eps=$(s.ϵ)")
        println(io, "dt,err_Erho,err_sigma2,err_rho")
        writedlm(io, hcat(Δt_values, err_Eρ, err_σ², err_ρ), ',')
    end
    return filename, Δt_values, err_Eρ, err_σ², err_ρ
end

function load_result(filename)
    data = readdlm(filename, ',', Float64; comments=true, comment_char='#', header=true)[1]
    return data[:, 1], data[:, 2], data[:, 3], data[:, 4]   # Δt, E[ρ], σ², ρ
end

# results[(ϵ, order)] = (Δt, [err_Eρ, err_σ², err_ρ])
results = Dict{Tuple{Float64,Int},Tuple{Vector{Float64},Vector{Vector{Float64}}}}()

for ϵ in EPS_VALUES, order in (1, 2)
    if RERUN_TUCKER
        _, Δt, eE, eσ, eρ = convergence_run(ϵ, order; naive=(NAIVE && order == 1))
    else
        sx = make_settings(ϵ)
        tag = order == 1 ? (NAIVE ? "order1_naive_" : "order1_") : ""
        fn = "convergence_$(tag)$(sx.problem)_Nx$(sx.Nx)_nPN$(sx.nPN)_Nxi$(sx.Nxi)_Neta$(sx.Neta)_tEnd$(sx.tEnd)_eps$(sx.ϵ).csv"
        Δt, eE, eσ, eρ = load_result(fn)
    end
    results[(ϵ, order)] = (Δt, [eE, eσ, eρ])
end

# ---------------------------------- plotting --------------------------------
function plot_convergence!(axes, Δt_values, err1_list, err2_list, ylabel_strs)
    Δt_ref = Δt_values[1]
    offset = 0.7
    for (ax, err1, err2, ylabel_str) in zip(axes, err1_list, err2_list, ylabel_strs)
        ref_first  = offset * err1[1] .* (Δt_values ./ Δt_ref)
        ref_second = offset * err2[1] .* (Δt_values ./ Δt_ref).^2

        ax.loglog(Δt_values, err1,       "o-", label="parallel, order 1")
        ax.loglog(Δt_values, err2,       ">-", label="parallel, order 2")
        ax.loglog(Δt_values, ref_first,  "--", label="1st order")
        ax.loglog(Δt_values, ref_second, ":",  label="2nd order")
        ax.set_xlabel("h")
        ax.set_ylabel(ylabel_str)
        ax.legend()
        ax.grid(true, which="both", linestyle="--")
        ax.invert_xaxis()
    end
end

superscript(ϵ) = replace(string(round(Int, log10(ϵ))), "-" => "⁻", "1" => "¹", "2" => "²",
                         "3" => "³", "4" => "⁴", "5" => "⁵", "6" => "⁶", "7" => "⁷",
                         "8" => "⁸", "9" => "⁹", "0" => "⁰")
epslabel(ϵ) = "ϑ = 10" * superscript(ϵ)
quantities = ["E[ρ]", "σ²", "ρ"]

# one figure per ϵ
for ϵ in EPS_VALUES
    Δt1, e1 = results[(ϵ, 1)]
    _,   e2 = results[(ϵ, 2)]
    ylabels = ["L2 error $(q), $(epslabel(ϵ))" for q in quantities]
    fig, axs = subplots(1, 3, figsize=(15, 4))
    plot_convergence!(axs, Δt1, e1, e2, ylabels)
    fig.tight_layout()
    fig.savefig("convergence_all_trunctol$(ϵ).pdf", dpi=300)
end

# combined figure: rows = quantities, columns = ϵ values
fig3, axes3 = subplots(3, length(EPS_VALUES), figsize=(6 * length(EPS_VALUES), 12), squeeze=false)
for (j, ϵ) in enumerate(EPS_VALUES)
    Δt1, e1 = results[(ϵ, 1)]
    _,   e2 = results[(ϵ, 2)]
    ylabels = ["L2 error $(q), $(epslabel(ϵ))" for q in quantities]
    plot_convergence!(axes3[:, j], Δt1, e1, e2, ylabels)
    axes3[1, j].set_title(epslabel(ϵ), fontsize=13, fontweight="bold")
end
fig3.tight_layout()
fig3.savefig("convergence_all_combined.pdf", dpi=300)
show()