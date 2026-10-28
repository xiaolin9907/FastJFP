module MathCompExamples
using LinearAlgebra, Krylov, SpecialFunctions, FFTW, KrylovKit, Random
include("GeneralJFP.jl")
using .GeneralJFP
export linear_build, linear_solve, physical_values, exact_values, sdc_solve,
    eigen_solve, GeneralJFP, airy_rl_build, airy_rl_solve

function gmres_solve(op,b;rtol=1e-13,itmax=400,restart=80)
    w=Krylov.GmresWorkspace(length(b),length(b),typeof(b);memory=min(length(b),restart))
    Krylov.gmres!(w,op,b;restart=true,reorthogonalization=true,rtol,atol=0.,itmax,history=true)
    (;u=copy(w.x),iterations=w.stats.niter,solved=w.stats.solved,history=copy(w.stats.residuals),status=w.stats.status)
end
function airy_rl_build(n,epsilon,method;nodes=160)
    method==:contour || throw(ArgumentError("This standalone version provides the contour method only"))
    n<=2000 || throw(ArgumentError("Airy sizes above 2000 disabled by user resource constraint"))
    np=n+5;d=epsilon*complex(0,1)^1.5;mx=[-.25,1.,.25]
    rhs=zeros(ComplexF64,n+1);rhs[1]=1
    w=IntegralWorkspace(n,1.5,.5;alpha=.5,nodes)
    jc=zeros(ComplexF64,n+3);prod=zeros(ComplexF64,np)
    op=FunctionOperator(ComplexF64,n+1,(out,c)->begin
        integral!(jc,w,view(c,1:n));mulpoly!(prod,jc,mx)
        out[1]=sum(jc)+c[end]
        @inbounds for i in 1:n;out[i+1]=d*c[i]-prod[i];end
        @inbounds for i in 1:3;out[i+1]-=mx[i]*c[end];end
        out
    end)
    (;op,b=rhs,w,n,epsilon)
end
function airy_rl_solve(n,epsilon,method;nodes=160)
    method==:contour || throw(ArgumentError("This standalone version provides the contour method only"))
    s=airy_rl_build(n,epsilon,method;nodes)
    r=gmres_solve(s.op,s.b;rtol=1e-10,itmax=400,restart=80)
    q=zeros(ComplexF64,n+3);integral!(q,s.w,view(r.u,1:n));q[1]+=r.u[end]
    (;r...,q)
end

function linear_build(name,n,method;nodes=160)
    method==:contour || throw(ArgumentError("This standalone version provides the contour method only"))
    np=n+8;rhs=zeros(n)
    if name==:abel
        beta=.5;mus=(.5,);lower=1;rhs[1]=1
    elseif name==:variable
        beta=1/6;mus=(1/3,.5);lower=5
        f=polyv(9,2.0^1.5;n=np)+polyv(14,2.0^(7/3)*(gamma(2.5)/gamma(17/6)+gamma(17/6)/gamma(10/3));n=np)
        rhs.=f[1:n]
    elseif name==:bbo
        beta=.5;mus=(.5,1.);lower=2;rhs[1]=-1
    else
        error("Unknown equation")
    end
    m1=polyv(3,sqrt(2));m2=polyv(2,2.0^(1/3))
    ws=[IntegralWorkspace(np,mu,beta;nodes) for mu in mus]
    padded=zeros(np);a=zeros(np);b=zeros(np);c=zeros(np)
    op=FunctionOperator(Float64,n,(out,x)->begin
        fill!(padded,0);padded[1:n].=x
        integral!(a,ws[1],padded)
        if name==:abel
            @inbounds for i in 1:n;out[i]=x[i]+4a[i];end
        elseif name==:variable
            mulpoly!(b,a,m1);mulpoly!(c,padded,m2);integral!(a,ws[2],c)
            @inbounds for i in 1:n;out[i]=x[i]+a[i]+b[i];end
        else
            integral!(b,ws[2],padded)
            @inbounds for i in 1:n;out[i]=x[i]+a[i]+b[i];end
        end
        out
    end)
    (;op,b=rhs,n,beta,lower,post=name==:bbo ? ws[2] : nothing)
end
function linear_solve(name,n,method;nodes=160)
    method==:contour || throw(ArgumentError("This standalone version provides the contour method only"))
    s=linear_build(name,n,method;nodes)
    r=gmres_solve(s.op,s.b;rtol=2e-14,itmax=400)
    if name==:bbo
        q=zeros(n+8);integral!(q,s.post,r.u)
        q[1]+=1
    else
        q=copy(r.u)
    end
    (;r...,q)
end
function physical_values(name,c,zs)
    if name==:bbo
        w=IntegralWorkspace(length(c),1.,.5;nodes=320)
        q=zeros(length(c)+2);integral!(q,w,c);q[1]+=1
    else
        q=c
    end
    [clenshaw(q,z) for z in zs]
end
function exact_values(name,zs)
    v=(1 .+ zs)./2
    if name==:abel
        erfcx.(4sqrt(2).*v)
    elseif name==:variable
        2.0^1.5 .* v.^9
    else
        # Laplace factorization in sqrt(s); this form is regular at t=0.
        r=complex(-.5,sqrt(3)/2)
        [2real((.5-im/(2sqrt(3)))*erfcx(-r*sqrt(2)*x)) for x in v]
    end
end

function vals2coeffs(v)
    # Values are ordered from z=+1 to z=-1.
    c=FFTW.r2r(v,FFTW.REDFT00)/(length(v)-1);c[1]/=2;c[end]/=2;c
end
function coeffs2vals(c)
    v=copy(c);v[2:end-1]./=2;FFTW.r2r(v,FFTW.REDFT00)
end
function sdc_solve(n,method;nodes=160,rtol=1e-15,maxiter=1000)
    method==:contour || throw(ArgumentError("This standalone version provides the contour method only"))
    z=cospi.((0:n-1)/(n-1));t=2 .* ((1 .+ z)./2).^2
    forcing=sqrt.(t)-sqrt(pi).*t./2;exact=sqrt(pi).*t./2
    low=zeros(n,n)
    # Descending grid: only earlier physical times (higher indices) contribute.
    for i in 1:n-1,j in i+1:n
        low[i,j]=(sqrt(t[i]-t[j])-sqrt(max(0,t[i]-t[j-1])))/gamma(1.5)
    end
    w=IntegralWorkspace(n,.5,.5;nodes)
    a=zeros(n);u=(I-low)\(low*forcing) # original forward-Euler initializer
    history=NamedTuple[]
    for k in 0:maxiter
        fc=vals2coeffs(u+forcing)
        integral!(a,w,fc)
        r=coeffs2vals(a)-u
        push!(history,(iteration=k,residual=norm(r,Inf),error=norm(u-exact,Inf)))
        norm(r,Inf)<=rtol && return (;u,history,iterations=k,solved=true)
        k==maxiter && break
        u .+= UpperTriangular(I-low)\r
    end
    (;u,history,iterations=maxiter,solved=false)
end

function eigen_solve(n,method;nodes=160,full=false)
    method==:contour || throw(ArgumentError("This standalone version provides the contour method only"))
    full && throw(ArgumentError("This standalone version uses matrix-free Arnoldi"))
    w=IntegralWorkspace(n,1.5,1.5;alpha=.5,nodes)
    op=x->begin
        y=similar(x);integral!(y,w,x);total=sum(y);y.*=-1;y[1]+=total;y
    end
    x=randn(MersenneTwister(20260919),n).*exp.(-36 .* (0:n-1)/(.8n))
    vals,vecs,info=KrylovKit.eigsolve(op,x,7,:LM;ishermitian=false,krylovdim=40,tol=1e-13,maxiter=100)
    (;values=vals,vectors=vecs,numops=info.numops,numiter=info.numiter,converged=info.converged)
end

end
