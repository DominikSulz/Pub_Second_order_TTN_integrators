__precompile__

include("Problem.jl")
include("utils.jl")

using LinearAlgebra
using LegendrePolynomials
using QuadGK
using TensorToolbox
using PyCall
np = pyimport("numpy")

struct Solver
    # spatial grid of cell interfaces
    x::Array{Float64};

    # Solver settings
    settings::Settings;

    Rhs
    SourceC
    SourceU


    # constructor
    function Solver(settings)
        x = settings.x;
        nx = settings.NCells;
        Δx = settings.Δx;

        # set up spatial stencil matrices
        Dₓ = Tridiagonal(-ones(nx-1)./Δx/2.0,zeros(nx),ones(nx-1)./Δx/2.0) # central difference matrix
        Dₓₓ = Tridiagonal(ones(nx-1)./Δx/2.0,-ones(nx)./Δx,ones(nx-1)./Δx/2.0) # stabilization matrix

        # setup right hand side 
        Rhs = [];
        SourceU = []; # Tucker bases of source
        SourceC = []; # core tensors of source
        if settings.problem == "radiationUQ"
            println("Problem is radiationUQ")
            nξ = settings.Nxi;
            nΩ = settings.nPN

            # setup flux matrix
            γ = ones(settings.nPN);

            # setup γ vector
            γ = zeros(settings.nPN);
            for i = 1:settings.nPN
                n = i-1;
                γ[i] = 2/(2*n+1);
            end
            
            # setup flux matrix
            A = zeros(settings.nPN,settings.nPN)

            for i = 1:(settings.nPN-1)
                n = i-1;
                A[i,i+1] = (n+1)/(2*n+1)*sqrt(γ[i+1])/sqrt(γ[i]);
            end

            for i = 2:settings.nPN
                n = i-1;
                A[i,i-1] = n/(2*n+1)*sqrt(γ[i-1])/sqrt(γ[i]);
            end

            # setup Roe matrix
            S = eigvals(A)
            V = eigvecs(A)
            AbsA = V*abs.(diagm(S))*inv(V)

            #Compute diagonal of scattering matrix G
            G = Diagonal([0.0;ones(settings.nPN-1)]);
            σₛ = Diagonal(ones(nx)).*settings.σₛ;
            σₐ = Diagonal(ones(nx)).*settings.σₐ;
            ξ, w = gausslegendre(settings.Nxi);
            σₛξ = Diagonal(settings.σₛξ .* ξ);

            RhsTerm = [-Dₓ, A, Diagonal(ones(nξ))]
            push!(Rhs,RhsTerm)
            RhsTerm = [Dₓₓ, AbsA, Diagonal(ones(nξ))]
            #push!(Rhs,RhsTerm) # add to stabilize
            RhsTerm = [-σₐ, Diagonal(ones(nΩ)), Diagonal(ones(nξ))]
            push!(Rhs,RhsTerm)
            RhsTerm = [-σₛ, G, Diagonal(ones(nξ))]
            push!(Rhs,RhsTerm)
            RhsTerm = [Diagonal(ones(nx)), G, σₛξ]
            push!(Rhs,RhsTerm)
        else
            println("Problem is ", settings.problem)
            Inx = Diagonal(ones(nx))
            RhsTerm = [Dₓₓ, Inx, Inx]
            push!(Rhs,RhsTerm)
            RhsTerm = [Inx, Dₓₓ, Inx]
            push!(Rhs,RhsTerm) # add to stabilize
            RhsTerm = [Inx, Inx, Dₓₓ]
            push!(Rhs,RhsTerm)

            G = zeros(nx, nx, nx)
            n = length(x)
            r = 11  # Tucker rank

            # Factor matrices: G1[i,k] = exp(-k * x[i]^2)
            G1 = [exp(-k * x[i]^2) for i in 1:n, k in 1:r]
            G2 = deepcopy(G1)
            G3 = deepcopy(G1)

            # Core tensor: only superdiagonal C[k,k,k] = 10^(-(k-1))
            CG = zeros(r, r, r)
            for k in 1:r
                CG[k, k, k] = 10.0^(-(k-1))
            end
            push!(SourceC, CG)
            push!(SourceU, [G1, G2, G3])

        end
        new(x,settings, Rhs, SourceC, SourceU);
    end
end

function SetupIC(obj::Solver)
    u = zeros(obj.settings.NCells,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1] = 2.0/sqrt(2)*IC(obj.settings.xMid);
    return u;
end

function F(obj::Solver, Y::Array{Float64,3})
    rhs = zeros(size(Y));
    d = 3;
    for i = 1:length(obj.Rhs)
        rhs .+= ttm(Y,Matrix.(obj.Rhs[i]),collect(1:d))
    end

    for i = 1:length(obj.Source)
        rhs .+= ttm(obj.SourceC[i],Matrix.(obj.SourceU[i]),collect(1:d))
    end
    #return -ttm(Y, [Matrix(obj.Dₓ), Matrix(obj.A)], [1, 2]) .+ ttm(Y, [Matrix(obj.Dₓₓ), Matrix(obj.AbsA)], [1, 2]) .- ttm(Y, [Matrix(obj.σₐ)], [1]) .- ttm(Y, [Matrix(obj.σₛ), Matrix(obj.G)], [1,2]) .- ttm(Y, [Matrix(obj.G), Matrix(obj.σₛξ)], [2,3])
    return rhs;
end

function precomputeProjection(obj::Solver, U⁰::Vector{Matrix{Float64}})
    rhsProject = Vector{Matrix{Float64}}[]
    srcProject = Vector{Matrix{Float64}}[]
    d = length(U⁰)
    for j = 1:length(obj.Rhs)
        termProjected = Matrix{Float64}[]
        for l in 1:d
            push!(termProjected, U⁰[l]'*obj.Rhs[j][l]*U⁰[l])
        end
        push!(rhsProject,termProjected)
    end

    for i = 1:length(obj.SourceC)
        termProjected = Matrix{Float64}[]
        for l in 1:d
            push!(termProjected, U⁰[l]'*obj.SourceU[j][l])
        end
        push!(srcProject, termProjected)
    end

    return rhsProject, srcProject
end

function F(obj::Solver, i::Int, r::Vector{Int}, K⁰::Matrix{Float64}, U⁰::Vector{Matrix{Float64}}, Q, rhsProject::Vector{Vector{Matrix{Float64}}}=[], srcProject::Vector{Vector{Matrix{Float64}}}=[], precomputed::Bool=false)
    d = 3;
    noti = collect(1:d); deleteat!(noti, i);
    rhs = zeros(size(K⁰));
    Qtensor = matten(Q,i,r)
    
    if !precomputed
        for j = 1:length(obj.Rhs)
            RhsProjected = Matrix{Float64}[]
            for l in 1:d
                if l != i
                    push!(RhsProjected, U⁰[l]'*obj.Rhs[j][l]*U⁰[l])
                else
                    push!(RhsProjected, obj.Rhs[j][l]*K⁰) # not really needed
                end
            end
            rhs .+= obj.Rhs[j][i]*K⁰*(tenmat(ttm(Qtensor, RhsProjected[noti], noti), i)*Q')
        end

        for j = 1:length(obj.SourceC)
            SourceProjected = Matrix{Float64}[]
            for l in 1:d
                push!(SourceProjected, U⁰[l]'*obj.SourceU[j][l])
            end
            rhs .+= obj.SourceU[j][i]*tenmat(ttm(obj.SourceC[j], SourceProjected[noti], noti), i)*Q'
        end
    else
        for j = 1:length(obj.Rhs)
            rhs .+= obj.Rhs[j][i]*K⁰*(tenmat(ttm(Qtensor, rhsProject[j][noti], noti), i)*Q')
        end

        for j = 1:length(obj.SourceC)
            rhs .+= obj.SourceU[j][i]*tenmat(ttm(obj.SourceC[j], srcProject[j][noti], noti), i)*Q'
        end
    end
    return rhs
end

# right-hand side for C-step
function F(obj::Solver, U⁰::Vector{Matrix{Float64}}, C::Array, rhsProject::Vector{Vector{Matrix{Float64}}}=[], srcProject::Vector{Vector{Matrix{Float64}}}=[], precomputed::Bool=false)
    d = 3;
    rhs = zeros(size(C));
    
    if !precomputed
        for j = 1:length(obj.Rhs)
            RhsProjected = Matrix{Float64}[]
            for l in 1:d
                push!(RhsProjected, U⁰[l]'*obj.Rhs[j][l]*U⁰[l])
            end
            rhs .+= ttm(C, RhsProjected, collect(1:d))
        end

        for j = 1:length(obj.SourceC)
            SourceProjected = Matrix{Float64}[]
            for l in 1:d
                push!(SourceProjected, U⁰[l]'*obj.SourceU[j][l])
            end
            rhs .+= obj.SourceU[j][i]*tenmat(ttm(obj.SourceC[j], SourceProjected, collect(1:d)), i)*Q'
        end
    else
        for j = 1:length(obj.Rhs)
            rhs .+= ttm(C, rhsProject[j], collect(1:d))
        end

        for j = 1:length(obj.SourceC)
            rhs .+= tenmat(ttm(obj.SourceC[j], srcProject[j], collect(1:d)), i)*Q'
        end
    end
    return rhs
end

# update and augment the ith basis matrix
function Φ(i::Int, C⁰, U⁰::Vector{Matrix{Float64}}, Fᵢ::Function, Δt, d::Int, r::Vector{Int}, N::Vector{Int})
    rᵢ = size(U⁰[i],2) 
    Qᵀ, Sᵀ = np.linalg.qr(tenmat(C⁰, i)', mode="reduced"); S⁰ = Sᵀ'; 
    K⁰ = U⁰[i]*S⁰;
    FK = K -> Fᵢ(K,Qᵀ')
    K¹ = rk(K -> Fᵢ(K,Qᵀ'), K⁰, Δt)
    Û¹,_ = np.linalg.qr([U⁰[i] K¹], mode="reduced"); 
    Û¹ = Matrix(Û¹[:,1:2*rᵢ]); Û¹[:,1:rᵢ] = U⁰[i]; Ũ = Matrix(Û¹[:,(rᵢ+1):2*rᵢ]);
    return Û¹, Û¹'*U⁰[i], matten((Ũ' * K¹)*Qᵀ', i, r)
end

# update and augment the ith basis matrix with 2nd order integrator
function Φ2nd(i::Int, C⁰, M::Vector{Matrix{Float64}}, U⁰::Vector{Matrix{Float64}}, Fᵢ::Function, CF::Array, Δt, d::Int, r::Vector{Int}, N::Vector{Int})
    rᵢ = size(U⁰[i],2) 
    noti = collect(1:d); deleteat!(noti, i);
    Ĉ⁰ = ttm(C⁰, M[noti], noti)
    Qᵀ, Sᵀ = np.linalg.qr(tenmat(Ĉ⁰, i)', mode="reduced");
    Q̂ᵀ, _ = np.linalg.qr([Qᵀ tenmat(CF, i)'*M[i]], mode="reduced"); 
    S⁰ = Sᵀ'; 
    K⁰ = U⁰[i]*M[i]*S⁰*Qᵀ'*Q̂ᵀ;
    K¹ = rk(K -> Fᵢ(K,Q̂ᵀ'), K⁰, Δt)
    Û¹,_ = np.linalg.qr([U⁰[i] K¹], mode="reduced"); 
    Û¹ = Matrix(Û¹[:,1:2*rᵢ]); Û¹[:,1:rᵢ] = U⁰[i]; Ũ = Matrix(Û¹[:,(rᵢ+1):2*rᵢ]);
    return Û¹, Û¹'*U⁰[i], matten((Ũ' * K¹)*Q̂ᵀ', i, r)
end

# update and augment the ith basis matrix
function pre_augment(i::Int, C⁰, U⁰::Vector{Matrix{Float64}}, Fᵢ::Function, Δt, d::Int, r::Vector{Int}, N::Vector{Int})
    rᵢ = size(U⁰[i],2)
    noti = collect(1:d); deleteat!(noti, i);
    Qᵀ, Sᵀ = np.linalg.qr(tenmat(C⁰, i)', mode="reduced"); S⁰ = Sᵀ'; 
    K⁰ = U⁰[i]*S⁰;

    Û⁰,_ = np.linalg.qr([U⁰[i] Fᵢ(K⁰,Qᵀ')], mode="reduced"); 
    Û⁰ = Matrix(Û⁰[:,1:2*rᵢ]); Û⁰[:,1:rᵢ] = U⁰[i];
    return Û⁰, Û⁰'*U⁰[i]
end

# augment and update core tensor
function Ψ(C⁰, M::Vector{Matrix{Float64}}, F::Function, Δt, dimRange::Vector{Int})
    Ĉ⁰ = ttm(C⁰, M, dimRange)
    return rk(C -> F(C), Ĉ⁰, Δt)
end

# augment and update core tensor
function ΨParallel(C⁰, F::Function, Δt)
    return Δt * F(C⁰)
end

function TuckerIntegratorStep(obj::Solver, C⁰, U⁰::Vector{Matrix{Float64}}, Δt, d::Int, r::Vector{Int}, N::Vector{Int}, precompute::Bool=true)
    Û¹ = Matrix{Float64}[];
    M = Matrix{Float64}[];

    # precompute projections
    if precompute
        rhsProject, srcProject = precomputeProjection(obj, U⁰)
    else
        rhsProject = Vector{Matrix{Float64}}[]
        srcProject = Vector{Matrix{Float64}}[]
    end

    for i = 1:d
        Ûᵢ, Mᵢ, _ = Φ(i, C⁰, U⁰, (K,Q) -> F(obj,i,r,K,U⁰,Q,rhsProject,srcProject,precompute), Δt, d, r, N)
        push!(Û¹, Ûᵢ); push!(M, Mᵢ);
    end
    Ĉ¹ = Ψ(C⁰, M, C -> F(obj, Û¹, C, Vector{Matrix{Float64}}[], false), Δt, collect(1:d));
    Ĉ¹, Û¹, r = θ(obj, Ĉ¹, Û¹, d, 2 .* r);
    return Ĉ¹, Û¹, r
end

function ParallelTuckerIntegratorStep(obj::Solver, C⁰, U⁰::Vector{Matrix{Float64}}, Δt, d::Int, r::Vector{Int}, N::Vector{Int}, precompute::Bool=true, naive::Bool=false)
    Û¹ = Matrix{Float64}[];

    # precompute projections
    if precompute
        rhsProject, srcProject = precomputeProjection(obj, U⁰)
    else
        rhsProject = Vector{Matrix{Float64}}[]
        srcProject = Vector{Matrix{Float64}}[]
    end

    rField = []
    for i = 1:d
        push!(rField,1:r[i])
    end 

    # create tensor with double dimension
    Ĉ¹ = matten(zeros(Float64, 2*r[1], prod(2 .* r[2:end])),1,2 .* r)

    for i = 1:d
        Ûᵢ, _, C̃ᵢ = Φ(i, C⁰, U⁰, (K,Q) -> F(obj,i,r,K,U⁰,Q,rhsProject,srcProject,precompute), Δt, d, r, N)

        rhsᵢ = deepcopy(rhsProject)
        srcᵢ = deepcopy(srcProject)
        Ũᵢ = Ûᵢ[:,(r[i]+1):end];
        for l in eachindex(rhsᵢ)
            rhsᵢ[l][i] = Ũᵢ'*obj.Rhs[l][i]*U⁰[i]
        end

        for l in eachindex(srcᵢ)
            srcᵢ[l][i] = Ũᵢ'*obj.Rhs[l][i]*U⁰[i]
        end

        rFieldᵢ = deepcopy(rField); rFieldᵢ[i] = rField[i] .+ r[i]
        if naive
            Ĉ¹[CartesianIndices(Tuple(rFieldᵢ))] = ΨParallel(C⁰, C -> F(obj, U⁰, C, rhsᵢ, srcProject, true), Δt);
            # srcProject not checked
        else
            Ĉ¹[CartesianIndices(Tuple(rFieldᵢ))] = C̃ᵢ;
        end
        push!(Û¹, [U⁰[i] Ũᵢ]);
    end

    # precompute projections
    if !precompute
        rhsProject = Vector{Matrix{Float64}}[]
        srcProject = Vector{Matrix{Float64}}[]
    end

    Ĉ¹[CartesianIndices(Tuple(rField))] = Ψ(C⁰,Matrix{Float64}[],C -> F(obj, U⁰, C, rhsProject, srcProject, precompute), Δt, Int[]);
    Ĉ¹, Û¹, r = θ(obj, Ĉ¹, Û¹, d, 2 .* r);
    return Ĉ¹, Û¹, r
end

function ParallelTuckerIntegratorStep2ndOrder(obj::Solver, C⁰, U⁰::Vector{Matrix{Float64}}, Δt, d::Int, r::Vector{Int}, N::Vector{Int}, precompute::Bool=true)
    Û¹ = Matrix{Float64}[];
    Û⁰ = Matrix{Float64}[];

    # precompute projections
    if precompute
        rhsProject, srcProject = precomputeProjection(obj, U⁰)
    else
        rhsProject = Vector{Matrix{Float64}}[]
        srcProject = Vector{Matrix{Float64}}[]
    end

    # augment basis
    M = Vector{Matrix{Float64}}(undef, d)
    for i in 1:d
        Ûᵢ⁰, Mᵢ = pre_augment(i, C⁰, U⁰, (K,Q) -> F(obj,i,r,K,U⁰,Q,rhsProject,srcProject,precompute), Δt, d, r, N)
        push!(Û⁰, Matrix(Ûᵢ⁰))
        M[i] = Matrix(Mᵢ)
    end

    rField = []
    for i = 1:d
        push!(rField,1:r[i])
    end 

    # augment core tensor
    Ĉ⁰ = ttm(C⁰, M, [1,2,3])

    # precompute projections
    if precompute
        rhsProject, srcProject = precomputeProjection(obj, Û⁰)
    else
        rhsProject = Vector{Matrix{Float64}}[]
        srcProject = Vector{Matrix{Float64}}[]
    end

    CF = F(obj, Û⁰, Ĉ⁰, rhsProject, srcProject)

    # create tensor with double dimension
    Ĉ¹ = matten(zeros(Float64, 4*r[1], prod(4 .* r[2:end])),1,4 .* r)

    for i = 1:d
        Ûᵢ, _, C̃ᵢ = Φ2nd(i, C⁰, M, Û⁰, (K,Q) -> F(obj,i,2 .* r,K,Û⁰,Q,rhsProject,srcProject,precompute), CF, Δt, d, 2 .* r, N)

        rhsᵢ = deepcopy(rhsProject)
        Ũᵢ = Ûᵢ[:,(2*r[i]+1):end];
        for l in eachindex(rhsᵢ)
            rhsᵢ[l][i] = Ũᵢ'*obj.Rhs[l][i]*Û⁰[i]
        end

        rFieldᵢ = Vector{UnitRange{Int}}(undef, d)
        for j = 1:d
            rFieldᵢ[j] = 1:(2*r[j])
        end
        rFieldᵢ[i] = (2*r[i]+1):(4*r[i])   # special direction
        Ĉ¹[CartesianIndices(Tuple(rFieldᵢ))] = C̃ᵢ;
        push!(Û¹, [Û⁰[i] Ũᵢ]);
    end

    # precompute projections
    if !precompute
        rhsProject = Vector{Matrix{Float64}}[]
        srcProject = Vector{Matrix{Float64}}[]
    end

    rField = Vector{UnitRange{Int}}(undef, d)
    for j = 1:d
        rField[j] = 1:(2*r[j])
    end
    Ĉ¹[CartesianIndices(Tuple(rField))] = Ψ(Ĉ⁰,Matrix{Float64}[],C -> F(obj, Û⁰, C, rhsProject, srcProject, precompute), Δt, Int[]);
    Ĉ¹, Û¹, r = θ(obj, Ĉ¹, Û¹, d, 4 .* r);
    return Ĉ¹, Û¹, r
end

function TuckerIntegrator(obj::Solver)
    s = obj.settings;
    r = [s.r,s.r,s.r]

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = [s.NCells,s.nPN,s.Nxi];

    # Set up initial condition
    X = rand(s.NCells,r[1])
    V = rand(s.nPN,r[2])
    W = rand(s.Nxi,r[3])

    X[:,1] = 2.0/sqrt(2)*IC(s.xMid);
    V[1,1] = 1; V[2:end,1] .= zeros(s.nPN-1);
    W[:,1] .= 1;
    C⁰ = zeros(s.r,s.r,s.r); C⁰[1,1,1] = 1;

    X, Rₓ = np.linalg.qr(X, mode="reduced");
    V, Rᵥ = np.linalg.qr(V, mode="reduced");
    W, R = np.linalg.qr(W, mode="reduced");
    C⁰ = ttm(C⁰,[Rₓ,Rᵥ,R],[1, 2, 3])
    U⁰ = [X, V, W];

    rankInTime = zeros(4,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= r;

        C⁰, U⁰, r = TuckerIntegratorStep(obj, C⁰, U⁰, Δt, 3, r, N)
        
        t += Δt;

        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, ttm(C⁰,U⁰,[1,2,3]), rankInTime;
end

function ParallelTuckerIntegrator(obj::Solver, precompute::Bool=true, naive::Bool=false)
    s = obj.settings;
    r = [s.r,s.r,s.r]

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = [s.NCells,s.nPN,s.Nxi];

    # Set up initial condition
    X = rand(s.NCells,r[1])
    V = rand(s.nPN,r[2])
    W = rand(s.Nxi,r[3])

    X[:,1] = 2.0/sqrt(2)*IC(s.xMid);
    V[1,1] = 1; V[2:end,1] .= zeros(s.nPN-1);
    W[:,1] .= 1;
    C⁰ = zeros(s.r,s.r,s.r); C⁰[1,1,1] = 1;

    X, Rₓ = np.linalg.qr(X, mode="reduced");
    V, Rᵥ = np.linalg.qr(V, mode="reduced");
    W, R = np.linalg.qr(W, mode="reduced");
    C⁰ = ttm(C⁰,[Rₓ,Rᵥ,R],[1, 2, 3])
    U⁰ = [X, V, W];

    rankInTime = zeros(4,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= r;

        C⁰, U⁰, r = ParallelTuckerIntegratorStep(obj, C⁰, U⁰, Δt, 3, r, N, precompute, naive)

        t += Δt;

        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, ttm(C⁰,U⁰,[1,2,3]), rankInTime;
end

function ParallelTuckerIntegrator2ndOrder(obj::Solver)
    s = obj.settings;
    r = [s.r,s.r,s.r]

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = [s.NCells,s.nPN,s.Nxi];

    # Set up initial condition
    X = rand(s.NCells,r[1])
    V = rand(s.nPN,r[2])
    W = rand(s.Nxi,r[3])

    X[:,1] = 2.0/sqrt(2)*IC(s.xMid);
    V[1,1] = 1; V[2:end,1] .= zeros(s.nPN-1);
    W[:,1] .= 1;
    C⁰ = zeros(s.r,s.r,s.r); C⁰[1,1,1] = 1;

    X, Rₓ = np.linalg.qr(X, mode="reduced");
    V, Rᵥ = np.linalg.qr(V, mode="reduced");
    W, R = np.linalg.qr(W, mode="reduced");
    C⁰ = ttm(C⁰,[Rₓ,Rᵥ,R],[1, 2, 3])
    U⁰ = [X, V, W];

    rankInTime = zeros(4,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= r;

        C⁰, U⁰, r = ParallelTuckerIntegratorStep2ndOrder(obj, C⁰, U⁰, Δt, 3, r, N)

        t += Δt;

        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, ttm(C⁰,U⁰,[1,2,3]), rankInTime;
end

function truncate!(obj::Solver,U::Array{Float64,2},Cᵢ::Array{Float64,2},i::Int,r::Vector{Int})
    # Compute singular values of S and decide how to truncate:
    P,D,Q = svd(Cᵢ);
    rmax = -1;
    rMaxTotal = obj.settings.rMax;
    rMinTotal = obj.settings.rMin;

    tmp = 0.0;
    tol = obj.settings.ϵ;#*norm(D);

    rmax = Int(floor(size(D,1)/2));

    for j=1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]).^2);
        if tmp < tol
            rmax = j;
            break;
        end
    end

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal;
    end

    rmax = min(rmax,rMaxTotal);
    rmax = max(rmax,rMinTotal);

    r[i] = rmax;
    C = matten(Diagonal(D[1:rmax])*Q[:,1:rmax]',i,r);

    # return rank
    return U*P[:, 1:rmax], C, r;
end

function θ(obj::Solver, Ĉ¹, Û¹::Vector{Matrix{Float64}}, d::Int, r::Vector{Int})
    for i = 1:d
        Cᵢ = tenmat(Ĉ¹,i);
        Û¹[i], Ĉ¹, r = truncate!(obj,Û¹[i],Cᵢ,i,r)
    end
    return Ĉ¹, Û¹, r
end

function FillTensor(ten,r::Int)
    if size(ten,1) != r || size(ten,2) != r || size(ten,3) != r
        tmp = ten;
        ten = zeros(r,r,r)
        for i = 1:size(tmp,1)
            for j = 1:size(tmp,2)
                for k = 1:size(tmp,3)
                    ten[i,j,k] = tmp[i,j,k];
                end
            end
        end
    end
    return ten
end

function FillTensor(ten, r::Vector{Int})
    if size(ten,1) != r[1] || size(ten,2) != r[2] || size(ten,3) != r[3]
        tmp = ten;
        ten = zeros(r[1],r[2],r[3])
        for i = 1:size(tmp,1)
            for j = 1:size(tmp,2)
                for k = 1:size(tmp,3)
                    ten[i,j,k] = tmp[i,j,k];
                end
            end
        end
    end
    return ten
end

function FillMatrix(mat,r)
    if size(mat,2) != r
        tmp = mat;
        mat = zeros(size(tmp,1),r)
        for i = 1:size(tmp,2)
            mat[:,i] = tmp[:,i];
        end
    end
    return mat
end



# update and augment the ith basis matrix
function ΦNaive(i::Int, C⁰, U⁰::Vector{Matrix{Float64}}, Fᵢ::Function, Δt, d::Int, r::Vector{Int}, N::Vector{Int})
    noti = collect(1:d); deleteat!(noti, i);
    rᵢ = size(U⁰[i],2) 
    Qᵀ, Sᵀ = np.linalg.qr(tenmat(C⁰, i)', mode="reduced"); S⁰ = Sᵀ'; 
    V = tenmat(ttm(matten(Qᵀ',i,r), U⁰[noti], noti), i)'
    K⁰ = U⁰[i]*S⁰;
    K¹ = K⁰ + Δt*Fᵢ(matten(K⁰*V', i, N))*V;
    Û¹,_ = np.linalg.qr([K¹ U⁰[i]], mode="reduced"); Û¹ = Matrix(Û¹[:,1:2*rᵢ]);
    return Û¹, Û¹'*U⁰[i]
end

# augment and update core tensor
function ΨNaive(C⁰, Û¹::Vector{Matrix{Float64}}, M::Vector{Matrix{Float64}}, F::Function, Δt, d::Int)
    Ĉ⁰ = ttm(C⁰, M, collect(1:d))
    return Ĉ⁰ + Δt * ttm(F(ttm(Ĉ⁰, Û¹, collect(1:d))), Matrix.(transpose.(Û¹)), collect(1:d))
end

function TuckerIntegratorStepNaive(obj::Solver, C⁰, U⁰::Vector{Matrix{Float64}}, F::Function, Δt, d::Int, r::Vector{Int}, N::Vector{Int})
    Û¹ = Matrix{Float64}[];
    M = Matrix{Float64}[];
    for i = 1:d
        Ûᵢ, Mᵢ = ΦNaive(i, C⁰, U⁰, Y -> tenmat(F(Y), i), Δt, d, r, N)
        push!(Û¹, Ûᵢ); push!(M, Mᵢ);
    end
    Ĉ¹ = ΨNaive(C⁰, Û¹, M, Y -> F(Y), Δt, d);
    Ĉ¹, Û¹, r = θ(obj, Ĉ¹, Û¹, d, 2 .* r);
    return Ĉ¹, Û¹, r
end

function TuckerIntegratorNaive(obj::Solver)
    s = obj.settings;
    r = [s.r,s.r,s.r]

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = [s.NCells,s.nPN,s.Nxi];

    # Set up initial condition
    u = zeros(s.NCells,s.nPN,s.Nxi); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1,:] .= 2.0/sqrt(2)*IC(s.xMid);

    # obtain tensor representation
    TT = hosvd(u,reqrank=r);
    C⁰ = TT.cten; C⁰ = FillTensor(C⁰,r);
    X = TT.fmat[1]; X = FillMatrix(X,r[1]);
    V = TT.fmat[2]; V = FillMatrix(V,r[2]);
    W = TT.fmat[3]; W = FillMatrix(W,r[3]);
    U⁰ = [X, V, W];

    rankInTime = zeros(4,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= r;

        C⁰, U⁰, r = TuckerIntegratorStepNaive(obj, C⁰, U⁰, Y -> F(obj, Y), Δt, 3, r, N)

        t += Δt;

        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, ttm(C⁰,U⁰,[1,2,3]), rankInTime;
end
