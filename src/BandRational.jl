module BandRational
include("AiryRational.jl")
include("AbelRational.jl")
using LinearAlgebra, Krylov, BandedMatrices
const AR=AiryRational
const MC=AR.MathCompExamples
const G=MC.GeneralJFP
const AB=AbelRational
export rational_diagonals, airy_build, airy_aux_build, abel_build, linear_build, linear_total, solve!, airy_total, abel_total, MC, G, AR, AB

"""Selected upper diagonals of sum weights[j]*(A+poles[j]*I)^(-1).
Column j stores entries R[j-k,j] in row k+1. No full N by N matrix is formed.
The recurrence constructs coefficients; the final solve uses LAPACK band LU.
"""
function rational_diagonals(n,beta,poles,weights,width;factor_pole=0.)
    K=min(width,n-1);R=zeros(K+1,n);v=zeros(K+1)
    # Each row coefficient is shared by up to K+1 columns. Cache denominators
    # (not reciprocals) to preserve the original floating-point divisions.
    first=zeros(n);second=similar(first);diagonal=similar(first)
    @inbounds for (b,w) in zip(poles,weights)
        for i in 1:n
            first[i]=beta*i/b
            second[i]=beta*(i+1)/(2b)-.5
            diagonal[i]=beta*(i-1)/(2b)+(i==1 ? 1. : .5)
        end
        residue=w/b
        for j in 1:n
            for k in 0:min(K,j-1)
                i=j-k;si=i==1 ? 1. : .5
                rhs=k==0 ? si : k==2 ? -.5 : 0.
                k>=1 && (rhs-=first[i]*v[k])
                k>=2 && (rhs-=second[i]*v[k-1])
                v[k+1]=rhs/diagonal[i]
                R[k+1,j]+=residue*v[k+1]
            end
        end
    end
    if factor_pole>0
        b=factor_pole
        @inbounds for j in 1:n
            for k in min(K,j-1):-1:0
                i=j-k;R[k+1,j]=(i==1 ? 1. : .5)*R[k+1,j]-(k>=2 ? R[k-1,j]/2 : 0.)
            end
            for k in 0:min(K,j-1)
                i=j-k;rhs=R[k+1,j]
                k>=1 && (rhs-=beta*i*R[k,j])
                k>=2 && (rhs-=(beta*(i+1)-b)/2*R[k-1,j])
                R[k+1,j]=rhs/(beta*(i-1)/2+b*(i==1 ? 1. : .5))
            end
        end
    end
    R
end

function airy_diagonals(n,width;m=32,smin=-5.,smax=15.)
    h=(smax-smin)/(m-1);t=exp.(range(smin,smax;length=m))
    weights=2h/pi .* t;weights[1]/=2;weights[end]/=2
    rational_diagonals(n,.5,1.25 .+ t.^2,weights,width;factor_pole=2.)
end

"""Assemble the actual retained band of left*r(A)*right, optionally times a
boundary difference basis. Intermediate polynomial degrees are retained.
"""
function product_band(R,n,upper,lower,left,right;difference=false,border=false,
                      identity=1.,factor=1.,T::Type{BT}=Float64) where {BT}
    upper=min(upper,n-1);lower=min(lower,n-1)
    center=lower+upper+1;B=zeros(BT,2lower+upper+1,n)
    ld=length(left)-1;rd=length(right)-1;K=size(R,1)-1;nr=size(R,2)
    rv=zeros(2rd+4);first=zeros(BT,n)
    @inbounds for j in 0:n-1
        base=max(0,j-(difference && j>=1 ? (j==1 ? 1 : 2) : 0)-rd)
        last=min(nr-1,j+rd);fill!(rv,0.)
        for term in 1:(difference && j>=1 ? 2 : 1)
            mode=term==1 ? j : j==1 ? 0 : j-2;sgn=term==1 ? 1. : -1.
            for k in 0:rd
                for dest in (mode+k,abs(mode-k))
                    base<=dest<=last && (rv[dest-base+1]+=sgn*right[k+1]/2)
                end
            end
        end
        imin=max(0,j-upper);imax=min(n-1,j+lower)
        for i in max(0,imin-ld):last
            value=0.
            for col in max(i,base):last
                col-i<=K && (value+=R[col-i+1,col+1]*rv[col-base+1])
            end
            for k in 0:ld
                a=factor*(left[k+1]/2*value)
                for dest in (i+k,abs(i-k))
                    imin<=dest<=imax && (B[center+dest-j,j+1]+=a)
                end
            end
        end
        B[center,j+1]+=identity
        if difference && j>=1
            prev=j==1 ? 0 : j-2
            j-prev<=upper && (B[center+prev-j,j+1]-=identity)
        end
    end
    if border
        for i in 0:lower;first[i+1]=B[center+i,1];end
        B[:,1].=0;B[center,1]=-1
    end
    (;B,lower,upper,first)
end

function factor_band(b)
    factor,piv=LAPACK.gbtrf!(b.lower,b.upper,size(b.B,2),b.B)
    (;factor,piv,lower=b.lower,upper=b.upper,first=b.first,
      rhs=zeros(eltype(factor),size(factor,2)),storage_bytes=sizeof(factor)+sizeof(piv)+sizeof(b.first))
end

function airy_build(n,epsilon;upper=64,m=32,smax=15.,nodes=160,restart=80,max_n=2000)
    AR.check_airy_size(n,max_n,epsilon)
    d=epsilon*complex(0,1)^1.5
    R=airy_diagonals(n+2,upper+5;m,smax)
    pre=factor_band(product_band(R,n,upper,5,G.polyv(3,2.0^1.5),[-.25,1.,.25];
                   difference=true,border=true,factor=-inv(d),T=ComplexF64))
    full=AR.stable_airy_build(n+2,epsilon;nodes,max_n,padding=2)
    # The N guard above applies to the physical unknown, not two padding modes.
    xc=zeros(ComplexF64,n+2);jc=zeros(ComplexF64,n+5)
    op=G.FunctionOperator(ComplexF64,n+1,(out,u)->begin
        G.mulpoly!(xc,view(u,1:n),[-.25,1.,.25]);full.integral_stable!(jc,xc)
        @inbounds for i in 1:n;out[i]=u[i]-jc[i]/d;end
        out[1]-=u[end];out[end]=sum(view(u,1:n));out
    end)
    pre! = (out,b)->begin
        @inbounds for i in 1:n;pre.rhs[i]=b[i]-b[end]*pre.first[i];end
        LAPACK.gbtrs!('N',pre.lower,pre.upper,n,pre.factor,pre.piv,pre.rhs)
        out[1]=b[end]-pre.rhs[2]-(n>=3 ? pre.rhs[3] : 0.)
        @inbounds for i in 2:n;out[i]=pre.rhs[i]-(i+2<=n ? pre.rhs[i+2] : 0.);end
        out[end]=pre.rhs[1];out
    end
    b=zeros(ComplexF64,n+1);b[end]=1
    workspace=Krylov.FgmresWorkspace(n+1,n+1,Vector{ComplexF64};memory=min(n+1,restart))
    (;n,epsilon,d,op,pre,pre!,workspace,b,full,kind=:airy,upper,m,smax,
      integral! = (out,c)->full.integral_stable!(out,c))
end

"""Boundary row of the short rational integral, evaluated with library
triangular solves. No dense fractional matrix is constructed."""
function airy_boundary(n;m=32,smin=-5.,smax=15.)
    D=BandedMatrix(0=>zeros(n),1=>zeros(n-1),2=>zeros(n-2))
    work=zeros(n);out=zeros(n)
    rt! = (dest,c,b)->begin
        @inbounds for i in 1:n
            D[i,i]=.25*(i-1)/b+(i==1 ? 1. : .5)
            i<n && (D[i,i+1]=.5i/b)
            i+2<=n && (D[i,i+2]=.25*(i+1)/b-.5)
        end
        copyto!(work,c);ldiv!(LowerTriangular(transpose(D)),work)
        @inbounds for i in 1:n;dest[i]=((i==1 ? 1. : .5)*work[i]-(i>=3 ? work[i-2]/2 : 0.))/b;end
    end
    first=zeros(n);rt!(first,ones(n),2.);ell=zeros(n)
    h=(smax-smin)/(m-1)
    for j in 0:m-1
        t=exp(smin+j*h);w=2h/pi*t*(j==0 || j==m-1 ? .5 : 1.)
        rt!(out,first,1.25+t*t);ell.+=w.*out
    end
    ell.*2.0^1.5
end

"""Preserve the original auxiliary equation, eliminating its border through
a rank-one update of an N-dimensional band LU (Sherman--Morrison)."""
function airy_aux_build(n,epsilon;upper=64,m=32,smax=15.,nodes=160,restart=80,max_n=2000)
    AR.check_airy_size(n,max_n,epsilon);d=epsilon*im^1.5
    R=airy_diagonals(n,upper+5;m,smax)
    p=G.polyv(5,2.);p[1:4].-=G.polyv(3)
    pre=factor_band(product_band(R,n,upper,5,p,[1.];identity=d,factor=-2.0^1.5,T=ComplexF64))
    ell=airy_boundary(n;m,smax);x0=zeros(ComplexF64,n);x0[1:3].=[-.25,1.,.25]
    y=copy(x0);LAPACK.gbtrs!('N',pre.lower,pre.upper,n,pre.factor,pre.piv,y)
    denominator=1+dot(ell,y)
    isfinite(denominator) && abs(denominator)>eps() || error("Unstable boundary Schur complement")
    pre! = (out,b)->begin
        @inbounds for i in 1:n;pre.rhs[i]=b[i+1]+x0[i]*b[1];end
        LAPACK.gbtrs!('N',pre.lower,pre.upper,n,pre.factor,pre.piv,pre.rhs)
        c=dot(ell,pre.rhs)/denominator
        @inbounds for i in 1:n;out[i]=pre.rhs[i]-y[i]*c;end
        out[end]=b[1]-dot(ell,view(out,1:n));out
    end
    full=AR.stable_airy_build(n,epsilon;nodes,max_n)
    workspace=Krylov.FgmresWorkspace(n+1,n+1,Vector{ComplexF64};memory=min(n+1,restart))
    (;n,epsilon,d,op=full.op,pre,pre!,workspace,b=full.b,full,kind=:airy_aux,upper,m,smax,denominator,
      preconditioner_bytes=pre.storage_bytes+sizeof(pre.rhs)+sizeof(ell)+sizeof(x0)+sizeof(y),
      integral! = full.integral_stable!)
end

function abel_build(n,mu;alpha=0.,lambda=4.,upper=32,nodes=160,rational_nodes=16,restart=80)
    n>=3 && isfinite(lambda) && restart>=1 || throw(ArgumentError("Invalid solver parameters"))
    rule=AB.rational_rule(mu;alpha,nodes=rational_nodes)
    R=rational_diagonals(n,mu,-rule.shifts,rule.weights,upper+1)
    pre=factor_band(product_band(R,n,upper,1,[.5,.5],[1.];factor=lambda))
    # Construct just the accurate action. AB.build_solver also allocates a
    # separate GMRES workspace and solve vectors unused by this FGMRES solver.
    plan=AB.PaperContour.action_plan(n,mu;alpha,nodes)
    action=AB.PaperContour.ContourApplyWorkspace(plan,1)
    op=AB.JFP.InplaceFunctionOperator(n,(y,x)->begin
        AB.PaperContour.contour_apply!(reshape(y,:,1),action,reshape(x,:,1))
        @. y=x+lambda*y
        y
    end)
    pre! = (out,b)->begin
        copyto!(pre.rhs,b);LAPACK.gbtrs!('N',pre.lower,pre.upper,n,pre.factor,pre.piv,pre.rhs)
        copyto!(out,pre.rhs)
    end
    workspace=Krylov.FgmresWorkspace(n,n,Vector{Float64};memory=min(n,restart))
    (;n,mu,alpha,lambda,op,pre,pre!,workspace,b=zeros(n),kind=:abel,upper)
end

function linear_build(name,n;upper=32,nodes=160,restart=80)
    full=MC.linear_build(name,n,:contour;nodes)
    terms = name==:abel ? [(mu=.5,left=G.polyv(1,4sqrt(2.)),right=[1.])] :
        name==:variable ? [(mu=1/3,left=G.polyv(5,2.0^(5/6)),right=[1.]),
                           (mu=.5,left=G.polyv(3,sqrt(2.)),right=G.polyv(2,2.0^(1/3)))] :
                          [(mu=.5,left=G.polyv(1,sqrt(2.)),right=[1.]),
                           (mu=1.,left=G.polyv(2,2.),right=[1.])]
    band=nothing
    for (k,term) in enumerate(terms)
        if term.mu==1.
            R=rational_diagonals(n+8,full.beta,[1.],[1.],upper+5)
        else
            rule=AB.rational_rule(term.mu)
            R=rational_diagonals(n+8,full.beta,-rule.shifts,rule.weights,upper+5)
        end
        b=product_band(R,n,upper,full.lower,term.left,term.right;identity=k==1 ? 1. : 0.)
        if band===nothing;band=b;else;band.B.+=b.B;end
    end
    pre=factor_band(band)
    pre! = (out,b)->begin
        copyto!(pre.rhs,b);LAPACK.gbtrs!('N',pre.lower,pre.upper,n,pre.factor,pre.piv,pre.rhs)
        copyto!(out,pre.rhs)
    end
    workspace=Krylov.FgmresWorkspace(n,n,Vector{Float64};memory=min(n,restart))
    (;n,op=full.op,pre,pre!,workspace,b=full.b,kind=:linear,upper,full,name)
end
function linear_total(name,n;rtol=2e-14,itmax=400,kwargs...)
    p=linear_build(name,n;kwargs...);r=solve!(p;rtol,itmax)
    if name==:bbo
        q=zeros(n+8);G.integral!(q,p.full.post,r.u);q[1]+=1
        return merge(r,(q=q,))
    end
    r
end

function solve!(s;b=s.b,rtol=1e-11,itmax=200,timemax=Inf,refinements=0,correction_rtol=1e-3)
    prior=get_zero_subnormals();set_zero_subnormals(true)
    try
        n=length(b);inverse=G.FunctionOperator(eltype(b),n,s.pre!)
        u=zeros(eltype(b),n);res=copy(b);hist=Float64[];stages=NamedTuple[];iterations=0
        normb=max(norm(b),eps(Float64));elapsed=0.
        for stage in 0:refinements
            tol=stage==0 ? rtol : correction_rtol
            elapsed+=@elapsed Krylov.fgmres!(s.workspace,s.op,res;N=inverse,restart=true,
                reorthogonalization=true,rtol=tol,atol=0.,itmax,timemax,history=true)
            u.+=s.workspace.x;append!(hist,s.workspace.stats.residuals)
            iterations+=s.workspace.stats.niter
            mul!(res,s.op,u);res.=b.-res
            rr=norm(res)/normb
            push!(stages,(stage,iterations=s.workspace.stats.niter,internal_solved=s.workspace.stats.solved,residual=rr))
            rr<=rtol && break
        end
        q=if s.kind==:airy_aux
            v=zeros(ComplexF64,s.n+3);s.integral!(v,view(u,1:s.n));v[1]+=u[end];v
        elseif s.kind==:airy
            copy(view(u,1:s.n))
        else
            copy(u)
        end
        (;u,q,iterations,
          solved=norm(res)/normb<=rtol,residual=norm(res)/normb,history=hist,stages,
          internal_solved=s.workspace.stats.solved,diagnostic_solve_seconds=elapsed)
    finally
        set_zero_subnormals(prior)
    end
end
"""Complete Airy solve using the original auxiliary equation by default.
Set formulation=:physical only for the separately tested diagnostic formulation.
"""
function airy_total(n,epsilon;rtol=1e-11,itmax=200,refinements=3,formulation=:auxiliary,kwargs...)
    formulation in (:auxiliary,:physical) || throw(ArgumentError("Unknown Airy formulation"))
    builder=formulation==:auxiliary ? airy_aux_build : airy_build
    solve!(builder(n,epsilon;kwargs...);rtol,itmax,refinements)
end
function abel_total(n,mu,b;rtol=1e-11,itmax=400,kwargs...)
    solve!(abel_build(n,mu;kwargs...);b,rtol,itmax)
end
end
