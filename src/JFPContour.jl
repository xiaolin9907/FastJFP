module JFPContour
using LinearAlgebra, Krylov
include("ContourJFP.jl")
include("ReferenceJFP.jl")
include("InplaceContour.jl")
export ContourJFP, ReferenceJFP, ContourApplyWorkspace, contour_apply!, build_abel, solve_abel!, fast_total, dense_total, structured_total, structured_matrix

function build_abel(n,mu;alpha=0.0,nodes=80,lambda=4.0,rtol=1e-14,rhs_drop_tol=1e-100)
    0<=rhs_drop_tol<=eps(Float64) || throw(ArgumentError("RHS cutoff must be at most machine epsilon"))
    plan=ContourJFP.ActionPlan(n,mu;alpha,nodes)
    action=ContourApplyWorkspace(plan,1)
    op=InplaceFunctionOperator(n,(y,x)->begin
        contour_apply!(reshape(y,:,1),action,reshape(x,:,1))
        @. y=x+lambda*y
        y
    end)
    workspace=Krylov.GmresWorkspace(n,n,Vector{Float64};memory=min(n,80))
    identity=InplaceFunctionOperator(n,(y,x)->copyto!(y,x))
    Krylov.gmres!(workspace,identity,zeros(n);restart=true,atol=0.0,rtol)
    (;op,workspace,plan,rtol,rhs_drop_tol,rhs=zeros(n),solution=zeros(n))
end
function solve_abel!(s,b)
    length(b)==s.plan.n || throw(DimensionMismatch())
    scale=prepare_rhs!(s.rhs,b,s.rhs_drop_tol)
    Krylov.gmres!(s.workspace,s.op,s.rhs;restart=true,reorthogonalization=true,
        atol=0.0,rtol=s.rtol,itmax=4length(b))
    s.workspace.stats.solved || error("GMRES did not converge")
    @. s.solution=scale*s.workspace.x
    s.solution
end
function prepare_rhs!(rhs,b,drop_tol)
    scale=maximum(abs,b)
    isfinite(scale) || throw(ArgumentError("Nonfinite right-hand side"))
    # Normalize before discarding coefficients 100 decades below the data
    # scale. The operator and N-dimensional Krylov problem remain linear and
    # unchanged; only negligible RHS entries are set to exact zero.
    @inbounds for i in eachindex(b)
        value=scale==0 ? 0.0 : b[i]/scale
        rhs[i]=abs(value)<drop_tol ? 0.0 : value
    end
    scale
end
fast_total(n,mu,b;kwargs...)=solve_abel!(build_abel(n,mu;kwargs...),b)
function dense_total(n,mu,b;alpha=0.0,lambda=4.0)
    A=lambda*ReferenceJFP.complete_jfp(n+1,mu;alpha)[1:n,1:n]
    for i in 1:n;A[i,i]+=1;end
    lu!(A)\b
end

"""Complete matrix with its exact upper-Hessenberg structure."""
function structured_matrix(n,mu;alpha=0.0,lambda=4.0)
    A=lambda*ReferenceJFP.complete_jfp(n+1,mu;alpha)[1:n,1:n]
    # Assert exact structure before changing its storage representation.
    for j in 1:n,i in j+2:n
        iszero(A[i,j]) || error("Unexpected entries below the first subdiagonal")
    end
    for i in 1:n;A[i,i]+=1;end
    UpperHessenberg(A)
end
function structured_total(n,mu,b;alpha=0.0,lambda=4.0,rhs_drop_tol=1e-100)
    A=structured_matrix(n,mu;alpha,lambda)
    rhs=similar(b);scale=prepare_rhs!(rhs,b,rhs_drop_tol)
    # Julia's specialized UpperHessenberg ldiv! uses the standard-library
    # O(N^2) Givens RQ solver. Do not use generic backslash, which selects LU.
    ldiv!(A,rhs)
    rhs .*= scale
    rhs
end
end
