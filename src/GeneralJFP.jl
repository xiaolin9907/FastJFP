"""General commensurate orders on [-1,1], in t^alpha T_j(2t^beta-1).
The production quadrature has M=160 fixed nodes, independent of N.
This module leaves the frozen beta=mu implementation unchanged.
"""
module GeneralJFP
using LinearAlgebra, SpecialFunctions
include("ContourJFP.jl")
export IntegralWorkspace, integral!, mulv!, mulpoly!, polyv, FunctionOperator, clenshaw

function gamma_ratio_log(w,mu)
    m=floor(Int,mu);d=mu-m
    value=d==0 ? zero(w) : ContourJFP.loggamma_ratio(w,d)
    for j in 0:m-1
        value-=log(w+d+j)
    end
    value
end

struct IntegralWorkspace
    n::Int
    mu::Float64
    beta::Float64
    alpha::Float64
    k::Int
    shifts::Vector{ComplexF64}
    weights::Vector{ComplexF64}
    constant::Float64
    threshold::Float64
    inverse::Matrix{ComplexF64}
    # The second upper diagonal is affine in the row and the shift.
    # Keep its real row part only; inverse diagonals still reuse the full plan.
    upper2::Vector{Float64}
    pinverse::Vector{Float64}
    pupper2::Vector{Float64}
    input::Matrix{Float64}
    rhs::Matrix{Float64}
    ac::Matrix{Float64}
    total::Matrix{Float64}
    compensation::Matrix{Float64}
    solution::Matrix{ComplexF64}
    product::Matrix{Float64}
    temporary::Matrix{Float64}
end

function IntegralWorkspace(n,mu,beta;alpha=0.,nodes=160)
    k=round(Int,mu/beta)
    n>=1 && mu>0 && beta>0 && alpha>-1 && k>=1 && isapprox(k*beta,mu;rtol=1e-13) ||
        throw(ArgumentError("Require mu=k*beta>0 and alpha>-1"))
    ca=alpha<0 ? alpha+1 : alpha
    h=asinh(34/8)/(nodes-1);b=.5*tan(deg2rad(35.))
    zs=zeros(ComplexF64,nodes);ws=similar(zs)
    for j in 0:nodes-1
        y=j*h;w=8sinh(y);z=-.5cosh(w)+im*b*sinh(w)
        dz=(-.5sinh(w)+im*b*cosh(w))*8cosh(y)
        zs[j+1]=z
        ws[j+1]=(j==0 || j==nodes-1 ? .5 : 1.)*h/pi*dz*exp(gamma_ratio_log(1+ca+z,mu))
    end
    invd=zeros(ComplexF64,n,nodes);up=[beta*(i+1) for i in 1:n]
    for j in eachindex(zs),i in 1:n
        invd[i,j]=inv(beta*(i-1)/2-zs[j]*(i==1 ? 1. : .5))
    end
    pinv=alpha<0 ? [inv(beta*(i-1)/2+(1+alpha)*(i==1 ? 1. : .5)) for i in 1:n] : Float64[]
    pu=alpha<0 ? [(beta*(i+1)-(1+alpha))/2 for i in 1:n] : Float64[]
    g0=exp(real(gamma_ratio_log(complex(1+ca),mu)))
    IntegralWorkspace(n,mu,beta,alpha,k,zs,ws,g0,max(32.,2beta*(n-1)^2),invd,up,pinv,pu,
        zeros(n,2),zeros(n,2),zeros(n,2),zeros(n,2),zeros(n,2),zeros(ComplexF64,n,2),
        zeros(n+k,2),zeros(n+k,2))
end

"""Multiplication by v=(1+z)/2 in orthogonal coefficients, without aliasing."""
function mulv!(out,c)
    fill!(out,0)
    @inbounds for i in eachindex(c)
        i<=length(out) && (out[i]+=c[i]/2)
        i>1 && i-1<=length(out) && (out[i-1]+=c[i]/4)
        i<length(out) && (out[i+1]+=(i==1 ? .5 : .25)*c[i])
    end
    out
end
function polyv(k,scale=1.;n=k+1)
    a=zeros(n);a[1]=scale;b=similar(a)
    for j in 1:k
        mulv!(b,a);a,b=b,a
    end
    a
end
"""Finite Chebyshev polynomial multiplier; input and output must not alias."""
function mulpoly!(out,c,p)
    fill!(out,0)
    @inbounds for j in eachindex(c),k in eachindex(p)
        a=j+k-1;b=abs(j-k)+1
        a<=length(out) && (out[a]+=c[j]*p[k]/2)
        b<=length(out) && (out[b]+=c[j]*p[k]/2)
    end
    out
end

"""Apply the original [-1,x] integral, optionally retaining all n+k modes.
Complex data use two real columns with the same cached resolvents.
"""
function integral!(out::AbstractVector,w::IntegralWorkspace,c::AbstractVector)
    n=w.n;length(c)<=n && length(out)<=n+w.k || throw(DimensionMismatch())
    C=w.input;b=eltype(c)<:Real ? 1 : 2
    fill!(C,0)
    @inbounds for i in eachindex(c)
        C[i,1]=real(c[i]);b==2 && (C[i,2]=imag(c[i]))
    end
    rhs=w.rhs;ac=w.ac;total=w.total;comp=w.compensation;sol=w.solution
    beta=w.beta
    @inbounds for col in 1:b,i in 1:n
        rhs[i,col]=(i==1 ? 1. : .5)*C[i,col]
        i+2<=n && (rhs[i,col]-=C[i+2,col]/2)
        ac[i,col]=beta*(i-1)/2*C[i,col]
        i+1<=n && (ac[i,col]+=beta*i*C[i+1,col])
        i+2<=n && (ac[i,col]+=beta*(i+1)/2*C[i+2,col])
        total[i,col]=w.constant*C[i,col];comp[i,col]=0
    end
    @inbounds for k in eachindex(w.shifts)
        z=w.shifts[k];weight=w.weights[k];factored=abs(z)>=w.threshold;iz=inv(z)
        for col in 1:b
            for i in n:-1:1
                val=factored ? ac[i,col]*iz : complex(rhs[i,col])
                i+1<=n && (val-=beta*i*sol[i+1,col])
                i+2<=n && (val-=(w.upper2[i]+z)/2*sol[i+2,col])
                sol[i,col]=val*w.inverse[i,k]
            end
            for i in 1:n
                correction=factored ? sol[i,col] : sol[i,col]+C[i,col]*iz
                term=imag(weight*correction)-comp[i,col]
                updated=total[i,col]+term;comp[i,col]=(updated-total[i,col])-term
                total[i,col]=updated
            end
        end
    end
    if w.alpha<0
        @inbounds for col in 1:b
            for i in n:-1:1
                val=(i==1 ? 1. : .5)*total[i,col]
                i+2<=n && (val-=total[i+2,col]/2)
                i+1<=n && (val-=beta*i*rhs[i+1,col])
                i+2<=n && (val-=w.pupper2[i]*rhs[i+2,col])
                rhs[i,col]=val*w.pinverse[i]
            end
            for i in 1:n
                total[i,col]+=w.mu*rhs[i,col]
            end
        end
    end
    p=w.product;tmp=w.temporary;fill!(p,0)
    @views p[1:n,1:b].=total[:,1:b]
    for k in 1:w.k
        for col in 1:b
            mulv!(view(tmp,:,col),view(p,:,col))
        end
        p,tmp=tmp,p
    end
    scale=2.0^w.mu
    @inbounds for i in eachindex(out)
        out[i]=scale*(b==1 ? p[i,1] : complex(p[i,1],p[i,2]))
    end
    out
end

struct FunctionOperator{T,F}
    n::Int
    f!::F
end
FunctionOperator(::Type{T},n,f) where T=FunctionOperator{T,typeof(f)}(n,f)
Base.size(A::FunctionOperator)=(A.n,A.n)
Base.size(A::FunctionOperator,d::Int)=d<=2 ? A.n : 1
Base.eltype(::FunctionOperator{T}) where T=T
LinearAlgebra.mul!(y::AbstractVector,A::FunctionOperator,x::AbstractVector)=A.f!(y,x)
function clenshaw(c,x)
    a=zero(eltype(c));b=a
    for j in length(c):-1:2
        a,b=c[j]+2x*a-b,a
    end
    c[1]+x*a-b
end
end
