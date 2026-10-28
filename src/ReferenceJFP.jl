module ReferenceJFP
using LinearAlgebra, SparseArrays, BandedMatrices, SpecialFunctions
export complete_jfp

function initial_row(n,mu,beta)
    h=min(0.1,1/(n-1)); m=max(72,8*n-8)
    z=collect(-m:m).*h
    a=(pi/2).*sinh.(z)
    w=(1 .+ tanh.(a))./2
    dw=(pi/4).*cosh.(z)./cosh.(a).^2
    x1=2 .* (1 .- w.^(1/mu)./2).^beta .- 1
    x2=2^(1-beta).*w.^beta .- 1
    w1=dw.*h/mu
    w2=dw.*(2 .- w).^(mu-1).*h
    row=zeros(n)
    old1=ones(length(w)); old2=copy(old1)
    now1=copy(x1); now2=copy(x2)
    row[1]=sum(w1)+sum(w2)
    row[2]=dot(now1,w1)+dot(now2,w2)
    for k=3:n
        new1=2 .*x1.*now1.-old1; new2=2 .*x2.*now2.-old2
        row[k]=dot(new1,w1)+dot(new2,w2)
        old1,now1=now1,new1; old2,now2=now2,new2
    end
    row
end
"""Independent beta-integral endpoint data, normalized by Gamma(mu).
Power substitutions on the two half intervals remove both singular weights.
All polynomial evaluations use the Chebyshev recurrence.
"""
function weighted_initial_row(n,mu,beta,alpha)
    a=1+alpha;h=min(0.025,1/(2*(n-1)));m=ceil(Int,8/h)
    x1=zeros(2m+1);x2=similar(x1);w1=similar(x1);w2=similar(x1)
    log1pexp(x)=max(x,0)+log1p(exp(-abs(x)))
    for (i,j) in enumerate(-m:m)
        z=j*h;v=pi*sinh(z)
        logw=-log1pexp(-v);logcw=-log1pexp(v)
        logdw=log(pi)+log(cosh(z))+logw+logcw+log(h)-loggamma(mu)
        logr1=log1p(-0.5*exp(logw/mu))
        logr2=-log(2.0)+logw/a
        x1[i]=2exp(beta*logr1)-1;x2[i]=2exp(beta*logr2)-1
        w1[i]=exp(logdw-mu*log(2.0)-log(mu)+alpha*logr1)
        w2[i]=exp(logdw-a*log(2.0)-log(a)+(mu-1)*log1p(-exp(logr2)))
    end
    old1=ones(length(x1));old2=copy(old1);now1=copy(x1);now2=copy(x2)
    row=zeros(n);row[1]=sum(w1)+sum(w2);row[2]=dot(now1,w1)+dot(now2,w2)
    for k in 3:n
        new1=2 .*x1.*now1.-old1;new2=2 .*x2.*now2.-old2
        row[k]=dot(new1,w1)+dot(new2,w2)
        old1,now1=now1,new1;old2,now2=now2,new2
    end
    row
end

function complete_jfp(n::Int,mu::Real;beta=mu,alpha=0.0)
    @assert n>=3
    mu=Float64(mu);beta=Float64(beta);alpha=Float64(alpha);k=round(Int,mu/beta)
    isfinite(alpha) && alpha>-1 || throw(ArgumentError("alpha must be finite and greater than -1"))
    0<mu<1 && beta>0 && k>=1 && isapprox(mu,k*beta) || throw(ArgumentError("Require 0<mu<1 and mu=k*beta"))
    R=zeros(n+2,n)
    if alpha==0
    m0=2^mu*gamma(mu)/gamma(1+mu)
    m1=2^(mu+beta)*gamma(mu)*gamma(1+beta)/gamma(1+mu+beta)
    m2=2^(mu+2*beta)*gamma(mu)*gamma(1+2*beta)/gamma(1+mu+2*beta)
    R[1,1]=m0
    R[2,2]=m1/2^beta; R[1,2]=m1/2^beta-m0
    R[3,3]=m2/2^(2*beta)
    R[2,3]=2^(2-2*beta)*m2-2^(2-beta)*m1
    R[1,3]=3*m2/2^(2*beta)-2^(2-beta)*m1+m0
    else
        g(j)=exp(loggamma(1+alpha+j*beta)-loggamma(1+alpha+j*beta+mu))
        g0=g(0);g1=g(1);g2=g(2)
        R[1,1]=g0
        R[2,2]=g1;R[1,2]=g1-g0
        R[3,3]=g2;R[2,3]=4g2-4g1;R[1,3]=3g2-4g1+g0
    end
    s=n+2
    T=spdiagm(s+1,s,0=>-ones(s),-1=>ones(s))
    MD=spdiagm(s,s+1,0=>beta.*collect(0:s-1)./2,
        1=>beta.*collect(1:s),2=>beta.*collect(2:s)./2)
    C=spdiagm(0=>vcat(1.0,fill(0.5,s-1)),2=>fill(-0.5,s-2))
    MT=MD*T; CT=C*T[1:s,:]; D=MD[:,1:s]
    row=alpha==0 ? initial_row(n,mu,beta) : weighted_initial_row(n,mu,beta,alpha)
    for j=4:n
        rhs=(2*j-2)*beta.*(C*R[:,j-1])+beta*(j-1).*(C*R[:,j-2])+
            ((j-1)/(j-3)).*(D*R[:,j-2])
        rhs[1]+=beta*(j-1)*row[j]
        op=BandedMatrix(MT-beta*(j-1)*CT,(1,2))
        v=op\rhs
        R[:,j]=T[1:s,:]*v
        R[1,j]+=row[j]
    end
    V=spdiagm(0=>fill(0.5,n),1=>fill(0.25,n-1),-1=>vcat(0.5,fill(0.25,n-2)))
    J=R[1:n,:]
    for _=1:k; J=V*J; end
    alpha==0 ? J.*(2.0^(-mu)/gamma(mu)) : J
end
end
