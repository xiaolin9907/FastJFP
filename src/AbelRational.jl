module AbelRational
include("PaperContour.jl")
using .PaperContour, LinearAlgebra, SparseArrays, SpecialFunctions, Krylov
const JFP = PaperContour.JFPContour
export rational_rule, rational_preconditioner, build_solver, solve!, total_solve, PaperContour

"""Short positive real-pole approximation to Gamma(z+alpha+1)/Gamma(z+alpha+mu+1).
Nodes are independent of N, lambda, and the right-hand side. Used only for preconditioning.
"""
function rational_rule(mu::Real;alpha=0.,nodes=16,smin=-5.,smax=9.)
    mu=Float64(mu);alpha=Float64(alpha)
    0<mu<1 && isfinite(alpha) && alpha>-1 || throw(ArgumentError("Require 0<mu<1 and alpha>-1"))
    nodes>=2 && isfinite(smin) && isfinite(smax) && smin<smax || throw(ArgumentError("Invalid quadrature"))
    shifted_alpha=alpha<0 ? alpha+1 : alpha
    delta=shifted_alpha+(1+mu)/2
    h=(smax-smin)/(nodes-1)
    poles=Float64[];weights=Float64[]
    for j in 0:nodes-1
        s=smin+j*h
        push!(poles,delta+exp(2s))
        push!(weights,2sinpi(mu)/pi*h*exp(2(1-mu)*s)*(j==0 || j==nodes-1 ? .5 : 1.))
    end
    # Match the exactly known constant-mode Gamma value, without fitting any solution.
    weights .*= exp(loggamma(1+shifted_alpha)-loggamma(1+shifted_alpha+mu))/sum(weights./poles)
    if alpha<0
        extra=0.;a=1+alpha
        for j in eachindex(weights)
            correction=mu*weights[j]/(poles[j]-a)
            weights[j]-=correction;extra+=correction
        end
        push!(poles,a);push!(weights,extra)
    end
    (;mu,alpha,shifts=-poles,weights,nodes,smin,smax)
end

function rational_preconditioner(n::Int,mu::Real;alpha=0.,lambda=4.,nodes=16,diagnostics=false)
    n>=3 && isfinite(lambda) || throw(ArgumentError("Invalid N or lambda"))
    r=rational_rule(mu;alpha,nodes);ns=length(r.shifts);block=ns+2;total=block*n
    idx(i,k)=(i-1)*block+k # original vector, summed rational action, shifted vectors
    ii=Int[];jj=Int[];vv=Float64[]
    sizehint!(ii,8ns*n);sizehint!(jj,8ns*n);sizehint!(vv,8ns*n)
    add(i,j,v)=iszero(v) ? nothing : (push!(ii,i);push!(jj,j);push!(vv,v);nothing)
    for i in 1:n
        add(idx(i,1),idx(i,1),1.)
        add(idx(i,1),idx(i,2),lambda/2)
        i>1 && add(idx(i-1,1),idx(i,2),lambda/4)
        i<n && add(idx(i+1,1),idx(i,2),lambda*(i==1 ? .5 : .25))
        add(idx(i,2),idx(i,2),1.)
        for j in 1:ns
            z=r.shifts[j];si=i==1 ? 1. : .5;pole=-z
            # Store pole*(A+pole*I)^(-1)u. This exact change of variables avoids
            # huge residues multiplying tiny auxiliary vectors in the sparse LU.
            add(idx(i,2),idx(i,j+2),-r.weights[j]/pole)
            add(idx(i,j+2),idx(i,1),-si)
            i+2<=n && add(idx(i,j+2),idx(i+2,1),.5)
            add(idx(i,j+2),idx(i,j+2),mu*(i-1)/(2pole)+si)
            i+1<=n && add(idx(i,j+2),idx(i+1,j+2),mu*i/pole)
            i+2<=n && add(idx(i,j+2),idx(i+2,j+2),mu*(i+1)/(2pole)-.5)
        end
    end
    P=sparse(ii,jj,vv,total,total)
    perm=[idx(i,k) for i in n:-1:1 for k in 1:block]
    fac=lu(P;q=perm)
    rhs=zeros(total);result=similar(rhs)
    apply! = (out,b)->begin
        fill!(rhs,0.)
        @inbounds for i in 1:n;rhs[idx(i,1)]=b[i];end
        ldiv!(result,fac,rhs)
        @inbounds for i in 1:n;out[i]=result[idx(i,1)];end
        out
    end
    (;rule=r,ns,total,nonzeros=nnz(P),factor_nonzeros=diagnostics ? nnz(fac.L)+nnz(fac.U) : -1,fac,apply!)
end

"""Both methods use the same M-node action and Krylov implementation. No coefficient chopping."""
function build_solver(n::Int,mu::Real;alpha=0.,lambda=4.,preconditioned=true,nodes=160,
                      rational_nodes=16,restart=80,rtol=1e-11,itmax=400,diagnostics=false)
    n>=3 && isfinite(lambda) && restart>=1 && rtol>0 && itmax>=1 || throw(ArgumentError("Invalid solver parameters"))
    plan=action_plan(n,mu;alpha,nodes)
    action=ContourApplyWorkspace(plan,1)
    original=JFP.InplaceFunctionOperator(n,(y,x)->begin
        contour_apply!(reshape(y,:,1),action,reshape(x,:,1))
        @. y=x+lambda*y
        y
    end)
    pc=preconditioned ? rational_preconditioner(n,mu;alpha,lambda,nodes=rational_nodes,diagnostics) : nothing
    temp=zeros(n)
    op=preconditioned ? JFP.InplaceFunctionOperator(n,(y,x)->begin
        pc.apply!(temp,x);mul!(y,original,temp);y
    end) : original
    workspace=Krylov.GmresWorkspace(n,n,Vector{Float64};memory=min(n,restart))
    (;n,mu=Float64(mu),alpha=Float64(alpha),lambda=Float64(lambda),preconditioned,plan,action,pc,
      original,op,workspace,rtol,itmax,rhs=zeros(n),solution=zeros(n),check=zeros(n))
end

function solve!(s,b)
    length(b)==s.n || throw(DimensionMismatch())
    all(isfinite,b) || throw(ArgumentError("Nonfinite RHS"))
    prior=get_zero_subnormals();set_zero_subnormals(true)
    try
        scale=maximum(abs,b);iszero(scale) && (scale=1.)
        @. s.rhs=b/scale
        Krylov.gmres!(s.workspace,s.op,s.rhs;restart=true,reorthogonalization=true,
            atol=0.,rtol=s.rtol,itmax=s.itmax,history=true)
        if s.preconditioned;s.pc.apply!(s.solution,s.workspace.x)
        else;copyto!(s.solution,s.workspace.x);end
        s.solution .*= scale
        mul!(s.check,s.original,s.solution)
        residual=norm(s.check-b)/(iszero(norm(b)) ? 1. : norm(b))
        (;u=copy(s.solution),iterations=s.workspace.stats.niter,solved=s.workspace.stats.solved,
          residual,history=copy(s.workspace.stats.residuals).*scale)
    finally
        set_zero_subnormals(prior)
    end
end

function total_solve(n,mu,b;kwargs...)
    prior=get_zero_subnormals();set_zero_subnormals(true)
    try
        solve!(build_solver(n,mu;kwargs...),b)
    finally
        set_zero_subnormals(prior)
    end
end
end
