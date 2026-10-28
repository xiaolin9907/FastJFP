"""Fixed global nodes for the Chebfun-type input experiments.

The default 160-node rule was optimized for mu=0.5, alpha=0 and a finite
tested N range. It is not a universal relative-error guarantee.
Load JFPContour.jl before including this module.
"""
module GlobalContour
using ..JFPContour
using SpecialFunctions, Krylov
export GlobalRule, fixed_plan, build_fixed_abel, fixed_total

struct GlobalRule
    mu::Float64
    alpha::Float64
    nodes::Int
    extent::Float64
    scale::Float64
    angle::Float64
    shifts::Vector{ComplexF64}
    weights::Vector{ComplexF64}
    constant::Float64
end

"""One global sinh grid and hyperbola, with no dependence on N.
angle is in radians. M counts upper-half-plane solves including the real
vertex; the conjugate half is incorporated by taking an imaginary part.
"""
function GlobalRule(mu::Real;alpha=0.,nodes=160,extent=34.,scale=8.,angle=deg2rad(35.))
    mu=Float64(mu);alpha=Float64(alpha)
    0<mu<1 && isfinite(alpha) && alpha>-1 || throw(ArgumentError("Invalid mu or alpha"))
    nodes>=2 && isfinite(extent) && extent>0 && isfinite(scale) && scale>0 && 0<angle<pi/2 ||
        throw(ArgumentError("Invalid global node parameters"))
    ca=alpha<0 ? alpha+1 : alpha
    h=asinh(extent/scale)/(nodes-1);b=.5*tan(angle)
    zs=Vector{ComplexF64}(undef,nodes);ws=similar(zs)
    for j in 0:nodes-1
        y=j*h;w=scale*sinh(y)
        z=-.5*cosh(w)+im*b*sinh(w)
        dz=(-.5*sinh(w)+im*b*cosh(w))*scale*cosh(y)
        zs[j+1]=z
        ws[j+1]=(j==0 || j==nodes-1 ? .5 : 1.)*h/pi*dz*
            exp(ContourJFP.loggamma_ratio(1+ca+z,mu))
    end
    all(isfinite,zs) && all(isfinite,ws) || throw(ArgumentError("Nonfinite contour data"))
    constant=alpha==0 ? exp(-loggamma(1+mu)) : exp(real(ContourJFP.loggamma_ratio(complex(1+ca),mu)))
    GlobalRule(mu,alpha,nodes,Float64(extent),Float64(scale),Float64(angle),zs,ws,constant)
end

"""Reuse exactly the same shifts and weights for every basis size."""
function fixed_plan(n::Int,r::GlobalRule)
    n>=1 || throw(ArgumentError("N must be positive"))
    mu=r.mu;alpha=r.alpha
    pinv=alpha<0 ? [inv(mu*(i-1)/2+(1+alpha)*(i==1 ? 1. : .5)) for i in 1:n] : Float64[]
    pup=alpha<0 ? [(mu*(i+1)-(1+alpha))/2 for i in 1:n] : Float64[]
    ContourJFP.ActionPlan(n,mu,alpha,r.nodes,36,r.shifts,r.weights,r.constant,
        max(32.,2mu*(n-1)^2),pinv,pup)
end

function build_fixed_abel(n,rule::GlobalRule;lambda=4.,rtol=1e-14,rhs_drop_tol=1e-100)
    0<=rhs_drop_tol<=eps(Float64) || throw(ArgumentError("Invalid RHS cutoff"))
    plan=fixed_plan(n,rule);action=ContourApplyWorkspace(plan,1)
    op=JFPContour.InplaceFunctionOperator(n,(y,x)->begin
        contour_apply!(reshape(y,:,1),action,reshape(x,:,1))
        @. y=x+lambda*y
        y
    end)
    workspace=Krylov.GmresWorkspace(n,n,Vector{Float64};memory=min(n,80))
    identity=JFPContour.InplaceFunctionOperator(n,(y,x)->copyto!(y,x))
    Krylov.gmres!(workspace,identity,zeros(n);restart=true,atol=0.,rtol)
    (;op,workspace,plan,rtol,rhs_drop_tol,rhs=zeros(n),solution=zeros(n))
end

"""End-to-end timing entry: includes node construction, all workspaces and solve."""
function fixed_total(n,mu,b;alpha=0.,nodes=160,extent=34.,scale=8.,angle=deg2rad(35.),lambda=4.,rtol=1e-14)
    rule=GlobalRule(mu;alpha,nodes,extent,scale,angle)
    solve_abel!(build_fixed_abel(n,rule;lambda,rtol),b)
end
end
