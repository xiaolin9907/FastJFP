module AiryRational
include("MathCompExamples.jl")
using .MathCompExamples, .MathCompExamples.GeneralJFP
using LinearAlgebra, SparseArrays, Krylov, BandedMatrices
export rational_build, rational_solve, solve!, stable_airy_build, MathCompExamples

"""Factor Gamma(z+3/2)/Gamma(z+3) into a real resolvent followed by
Gamma(z+3/2)/Gamma(z+2); reduces cancellation on Airy's large auxiliary vector.
"""
function check_airy_size(n,max_n,epsilon=NaN)
    # The larger cap is restricted to the three explicitly requested small-epsilon studies.
    cap=epsilon in (1e-7,1e-8,1e-9) ? 1000000 : 10000
    16<=n<=min(max_n,cap) || throw(ArgumentError(
        "Airy requires 16 <= N <= min(max_n, $cap); default max_n=2000. epsilon=1e-6, N=50000 remains prohibited."))
    nothing
end
function stable_airy_build(n,epsilon;nodes=160,max_n=2000,padding=0)
    0<=padding<=2 || throw(ArgumentError("Only zero to two internal padding modes are allowed"))
    check_airy_size(n-padding,max_n,epsilon)
    D=BandedMatrix(0=>[i==1 ? 2. : .25*(i-1)+1 for i in 1:n],
        1=>[.5i for i in 1:n-1],2=>[.25*(i+1)-1 for i in 1:n-2])
    w=IntegralWorkspace(n,.5,.5;alpha=.5,nodes)
    y=zeros(ComplexF64,n);h=zeros(ComplexF64,n+1)
    temp=zeros(ComplexF64,n+2);jc=zeros(ComplexF64,n+3);prod=zeros(ComplexF64,n+5)
    integral_stable! = (out,c)->begin
        @inbounds for i in 1:n
            y[i]=(i==1 ? 1. : .5)*c[i]-(i+2<=n ? c[i+2]/2 : 0)
        end
        ldiv!(UpperTriangular(D),y)
        integral!(h,w,y);mulv!(temp,h);mulv!(jc,temp)
        @inbounds for i in eachindex(out);out[i]=2jc[i];end
        out
    end
    d=epsilon*complex(0,1)^1.5;mx=[-.25,1.,.25];jout=similar(jc)
    op=FunctionOperator(ComplexF64,n+1,(out,c)->begin
        integral_stable!(jout,view(c,1:n));mulpoly!(prod,jout,mx)
        out[1]=sum(jout)+c[end]
        @inbounds for i in 1:n;out[i+1]=d*c[i]-prod[i];end
        @inbounds for i in 1:3;out[i+1]-=mx[i]*c[end];end
        out
    end)
    b=zeros(ComplexF64,n+1);b[1]=1
    (;op,b,w,n,epsilon,integral_stable!)
end

function rational_build(n,epsilon;m=16,endpoint=20.,stretch=4.,angle=35.,restart=80,rule=:sqrt,
                        smin=-5.,smax=9.,stable=true,diagnostics=false,ordering=:backward,max_n=2000,
                        scaled_aux=false,flexible=false)
    check_airy_size(n,max_n,epsilon)
    d=epsilon*complex(0,1)^1.5;scale=2.0^1.5
    h=asinh(endpoint/stretch)/(m-1);b=.5tan(deg2rad(angle))
    shifts=ComplexF64[];weights=ComplexF64[]
    for j in (rule==:contour ? (0:m-1) : (1:0))
        y=j*h;w=stretch*sinh(y)
        z=-.5cosh(w)+im*b*sinh(w)
        dz=(-.5sinh(w)+im*b*cosh(w))*stretch*cosh(y)
        a=(j==0 || j==m-1 ? .5 : 1.)*h/pi*dz*exp(GeneralJFP.gamma_ratio_log(1.5+z,1.5))
        push!(shifts,z,conj(z));push!(weights,a/(2im),-conj(a)/(2im))
    end
    # No constant term: preserve decay of the rational surrogate at infinity.
    if rule==:sqrt
        # Gamma(z+3/2)/Gamma(z+3) ~ [(z+2)*sqrt(z+5/4)]^(-1).
        # Positive sinc quadrature for the inverse square root; partial fractions.
        step=(smax-smin)/(m-1);r2=0.
        for j in 0:m-1
            t=exp(smin+j*step);pole=1.25+t*t
            weight=2step/pi*t*(j==0 || j==m-1 ? .5 : 1.)/(2-pole)
            push!(shifts,-pole);push!(weights,weight);r2-=weight
        end
        push!(shifts,-2.);push!(weights,r2)
    elseif rule==:contour
        weights .*= (sqrt(pi)/4)/real(sum(-weights./shifts))
    else
        error("unknown rational rule")
    end
    ns=length(shifts);block=ns+2;total=n*block+1
    idx(i,k)=(i-1)*block+k # k=1: v; k=2: q; k>=3: resolvents
    ii=Int[];jj=Int[];vv=ComplexF64[]
    add(i,j,v)=iszero(v) ? nothing : (push!(ii,i);push!(jj,j);push!(vv,v);nothing)
    # x*v^3 is a degree-five Chebyshev multiplier in the fractional coordinate.
    p=polyv(5,2.);p[1:4].-=polyv(3)
    for col in 1:n, k in eachindex(p)
        row=col+k-1
        row<=n && add(idx(row,1),idx(col,2),-scale*p[k]/2)
        row=abs(col-k)+1
        row<=n && add(idx(row,1),idx(col,2),-scale*p[k]/2)
    end
    for i in 1:n
        add(idx(i,1),idx(i,1),d)
        i<=3 && add(idx(i,1),total,-[-.25,1.,.25][i])
        add(idx(i,2),idx(i,2),1.)
        add(total,idx(i,2),scale)
        for k in 1:ns
            pole_scale=scaled_aux ? -real(shifts[k]) : 1.
            add(idx(i,2),idx(i,k+2),-weights[k]/pole_scale)
            si=i==1 ? 1. : .5
            add(idx(i,k+2),idx(i,1),-si)
            i+2<=n && add(idx(i,k+2),idx(i+2,1),.5)
            add(idx(i,k+2),idx(i,k+2),(.5*(i-1)/2-shifts[k]*si)/pole_scale)
            i+1<=n && add(idx(i,k+2),idx(i+1,k+2),.5i/pole_scale)
            i+2<=n && add(idx(i,k+2),idx(i+2,k+2),(.5*(i+1)+shifts[k])/(2pole_scale))
        end
    end
    add(total,total,1.)
    P=sparse(ii,jj,vv,total,total)
    if ordering==:natural
        fac=lu(P;q=collect(1:total))
    elseif ordering==:backward
        perm=vcat([idx(i,k) for i in n:-1:1 for k in 1:block],total)
        fac=lu(P;q=perm)
    else
        fac=lu(P)
    end
    rhs=zeros(ComplexF64,total);z=similar(rhs);u=zeros(ComplexF64,n+1)
    pre! = (out,y)->begin
        fill!(rhs,0);rhs[end]=y[1]
        @inbounds for i in 1:n;rhs[idx(i,1)]=y[i+1];end
        ldiv!(z,fac,rhs)
        @inbounds for i in 1:n;out[i]=z[idx(i,1)];end
        out[end]=z[end];out
    end
    full=stable ? stable_airy_build(n,epsilon;max_n) : airy_rl_build(n,epsilon,:contour)
    op=FunctionOperator(ComplexF64,n+1,(out,y)->begin
        pre!(u,y);mul!(out,full.op,u);out
    end)
    krylov=flexible ? Krylov.FgmresWorkspace(n+1,n+1,Vector{ComplexF64};memory=min(n+1,restart)) :
                      Krylov.GmresWorkspace(n+1,n+1,Vector{ComplexF64};memory=min(n+1,restart))
    (;n,epsilon,m,ns,shifts,weights,total,nonzeros=nnz(P),
        factor_nonzeros=diagnostics ? nnz(fac.L)+nnz(fac.U) : -1,fac,pre!,full,op,krylov,flexible)
end
function solve!(s;b=s.full.b,rtol=1e-11,itmax=200,timemax=Inf,callback=workspace->false)
    prior=get_zero_subnormals();set_zero_subnormals(true)
    try
        _solve!(s;b,rtol,itmax,timemax,callback)
    finally
        set_zero_subnormals(prior)
    end
end
function _solve!(s;b=s.full.b,rtol=1e-11,itmax=200,timemax=Inf,callback=workspace->false)
    if s.flexible
        pre=FunctionOperator(ComplexF64,s.n+1,s.pre!)
        Krylov.fgmres!(s.krylov,s.full.op,b;N=pre,restart=true,reorthogonalization=true,
                       rtol,atol=0.,itmax,timemax,callback,history=true)
        u=copy(s.krylov.x)
    else
        Krylov.gmres!(s.krylov,s.op,b;restart=true,reorthogonalization=true,rtol,atol=0.,itmax,timemax,callback,history=true)
        u=similar(b);s.pre!(u,s.krylov.x)
    end
    q=zeros(ComplexF64,s.n+3)
    if hasproperty(s.full,:integral_stable!)
        s.full.integral_stable!(q,view(u,1:s.n))
    else
        integral!(q,s.full.w,view(u,1:s.n))
    end
    q[1]+=u[end]
    tmp=similar(b);mul!(tmp,s.full.op,u)
    (;u,q,iterations=s.krylov.stats.niter,solved=s.krylov.stats.solved,
        history=copy(s.krylov.stats.residuals),residual=norm(tmp-b)/norm(b))
end
function rational_solve(n,epsilon;kwargs...)
    s=rational_build(n,epsilon;kwargs...);solve!(s)
end
end
