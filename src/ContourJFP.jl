module ContourJFP
using SpecialFunctions
export ActionPlan, apply, apply_transpose, default_nodes

"""Fixed contour resolution, independent of basis size; accuracy is checked numerically."""
default_nodes(::Integer)=80

function loggamma_ratio(w,mu)
    abs(w)<32 && return loggamma(w)-loggamma(w+mu)
    if real(w)<0
        s=sign(imag(w))
        sine_ratio=-s*im*pi*mu+log1p(-exp(s*2pi*im*(w+mu)))-log1p(-exp(s*2pi*im*w))
        return loggamma_ratio(1-w-mu,mu)+sine_ratio
    end
    value=zero(w);factor=1.0
    for k in 1:12
        factor*=mu/k
        value-=factor*polygamma(k-1,w)
    end
    value
end

struct ActionPlan
    n::Int
    mu::Float64
    alpha::Float64
    nodes::Int
    precision::Int
    shifts::Vector{ComplexF64}
    weights::Vector{ComplexF64}
    constant::Float64
    threshold::Float64
    pole_inverse::Vector{Float64}
    pole_upper2::Vector{Float64}
end

function ActionPlan(n::Int,mu::Real;alpha=0.0,nodes=default_nodes(n),precision=36)
    n>=1 && 0<mu<1 && nodes>=2 && precision>=4 || throw(ArgumentError("Invalid contour parameters"))
    mu=Float64(mu);alpha=Float64(alpha)
    isfinite(alpha) && alpha>-1 || throw(ArgumentError("The endpoint exponent alpha must be finite and greater than -1"))
    # For negative alpha, remove the nearest pole with the exact Gamma
    # recurrence. The contour then evaluates g_(mu,alpha+1), whose poles
    # remain strictly left of -1, even when alpha approaches -1.
    contour_alpha=alpha<0 ? alpha+1 : alpha
    bound=max(1.0,2mu*(n-1)^2)
    R=exp((log(bound)+precision)/(1+mu))
    W=asinh(sqrt((2R-1)*(2R+1))/sqrt(2.0));Y=asinh(2W/pi);h=Y/(nodes-1)
    zs=Vector{ComplexF64}(undef,nodes);ws=similar(zs)
    for j in 0:nodes-1
        y=j*h;w=pi/2*sinh(y);dw=pi/2*cosh(y)
        z=-cosh(w)/2+0.5im*sinh(w)
        dz=(-sinh(w)/2+0.5im*cosh(w))*dw
        zs[j+1]=z
        ws[j+1]=(j==0 ? 0.5 : 1.0)*h/pi*dz*exp(loggamma_ratio(1+contour_alpha+z,mu))
    end
    all(isfinite,zs) && all(isfinite,ws) || error("Nonfinite contour data")
    constant=alpha==0 ? exp(-loggamma(1+mu)) : exp(real(loggamma_ratio(complex(1+contour_alpha),mu)))
    pole_inverse=alpha<0 ? [inv(mu*(i-1)/2+(1+alpha)*(i==1 ? 1.0 : 0.5)) for i in 1:n] : Float64[]
    pole_upper2=alpha<0 ? [(mu*(i+1)-(1+alpha))/2 for i in 1:n] : Float64[]
    ActionPlan(n,mu,alpha,nodes,precision,zs,ws,constant,max(32.0,2mu*(n-1)^2),pole_inverse,pole_upper2)
end

"""Apply I + mu*(A+(1+alpha)I)^(-1), using a reusable real buffer.
Only needed for negative alpha; total holds polynomial coefficients, not
values of the possibly singular function t^alpha*p.
"""
function pole_correction!(total,p::ActionPlan,buffer;transposed=false)
    p.alpha<0 || return total
    n,b=size(total)
    if transposed
        # (A+aI)^(-T) = S0^T*(DA+a*S0)^(-T).
        @inbounds for col in 1:b
            for i in 1:n
                value=total[i,col]
                i>1 && (value-=p.mu*(i-1)*buffer[i-1,col])
                i>2 && (value-=p.pole_upper2[i-2]*buffer[i-2,col])
                buffer[i,col]=value*p.pole_inverse[i]
            end
            for i in 1:n
                value=(i==1 ? 1.0 : 0.5)*buffer[i,col]
                i>2 && (value-=buffer[i-2,col]/2)
                total[i,col]+=p.mu*value
            end
        end
    else
        @inbounds for col in 1:b
            for i in n:-1:1
                value=(i==1 ? 1.0 : 0.5)*total[i,col]
                i+2<=n && (value-=total[i+2,col]/2)
                i+1<=n && (value-=p.mu*i*buffer[i+1,col])
                i+2<=n && (value-=p.pole_upper2[i]*buffer[i+2,col])
                buffer[i,col]=value*p.pole_inverse[i]
            end
            for i in 1:n
                total[i,col]+=p.mu*buffer[i,col]
            end
        end
    end
    total
end

"""Action in t^alpha*T_j(2t^mu-1). full=true retains degree N."""
function apply(p::ActionPlan,C::AbstractMatrix{Float64};full=false)
    n,b=size(C);n==p.n || throw(DimensionMismatch())
    rhs=zeros(n,b);ac=zeros(n,b)
    @inbounds for col in 1:b,i in 1:n
        rhs[i,col]=(i==1 ? 1.0 : 0.5)*C[i,col]
        i+2<=n && (rhs[i,col]-=C[i+2,col]/2)
        ac[i,col]=p.mu*(i-1)/2*C[i,col]
        i+1<=n && (ac[i,col]+=p.mu*i*C[i+1,col])
        i+2<=n && (ac[i,col]+=p.mu*(i+1)/2*C[i+2,col])
    end
    total=p.constant.*C;compensation=zeros(n,b);solution=zeros(ComplexF64,n,b)
    inverse=zeros(ComplexF64,n);upper2=similar(inverse)
    @inbounds for k in eachindex(p.shifts)
        z=p.shifts[k];weight=p.weights[k];factored=abs(z)>=p.threshold
        for i in 1:n
            inverse[i]=inv(p.mu*(i-1)/2-z*(i==1 ? 1.0 : 0.5))
            upper2[i]=(p.mu*(i+1)+z)/2
        end
        iz=inv(z)
        for col in 1:b
            for i in n:-1:1
                value=factored ? ac[i,col]*iz : complex(rhs[i,col])
                i+1<=n && (value-=p.mu*i*solution[i+1,col])
                i+2<=n && (value-=upper2[i]*solution[i+2,col])
                solution[i,col]=value*inverse[i]
            end
            for i in 1:n
                correction=factored ? solution[i,col] : solution[i,col]+C[i,col]*iz
                term=imag(weight*correction)-compensation[i,col]
                updated=total[i,col]+term
                compensation[i,col]=(updated-total[i,col])-term
                total[i,col]=updated
            end
        end
    end
    pole_correction!(total,p,rhs)
    # Exact coefficient multiplication, followed by the requested projection.
    out=zeros(full ? n+1 : n,b)
    @inbounds for col in 1:b,i in 1:n
        out[i,col]+=total[i,col]/2
        i>1 && (out[i-1,col]+=total[i,col]/4)
        i<size(out,1) && (out[i+1,col]+=(i==1 ? 0.5 : 0.25)*total[i,col])
    end
    out
end
apply(p::ActionPlan,c::AbstractVector{Float64};kwargs...)=vec(apply(p,reshape(c,:,1);kwargs...))

"""Real transpose of the same finite action, using forward three-band solves."""
function apply_transpose(p::ActionPlan,C::AbstractMatrix{Float64})
    n,b=size(C);n==p.n || throw(DimensionMismatch())
    v=zeros(n,b)
    @inbounds for col in 1:b,i in 1:n
        v[i,col]=C[i,col]/2
        i>1 && (v[i,col]+=C[i-1,col]/4)
        i<n && (v[i,col]+=(i==1 ? 0.5 : 0.25)*C[i+1,col])
    end
    p.alpha<0 && pole_correction!(v,p,similar(v);transposed=true)
    total=p.constant.*v;compensation=zeros(n,b);y=zeros(ComplexF64,n,b)
    inverse=zeros(ComplexF64,n);lower2=similar(inverse)
    @inbounds for k in eachindex(p.shifts)
        z=p.shifts[k];weight=p.weights[k];factored=abs(z)>=p.threshold;iz=inv(z)
        for i in 1:n
            inverse[i]=inv(p.mu*(i-1)/2-z*(i==1 ? 1.0 : 0.5))
            lower2[i]=(p.mu*(i-1)+z)/2
        end
        for col in 1:b
            for i in 1:n
                value=complex(v[i,col])
                i>1 && (value-=p.mu*(i-1)*y[i-1,col])
                i>2 && (value-=lower2[i]*y[i-2,col])
                y[i,col]=value*inverse[i]
            end
            for i in 1:n
                if factored
                    correction=p.mu*(i-1)/2*y[i,col]
                    i>1 && (correction+=p.mu*(i-1)*y[i-1,col])
                    i>2 && (correction+=p.mu*(i-1)/2*y[i-2,col])
                    correction*=iz
                else
                    correction=(i==1 ? 1.0 : 0.5)*y[i,col]+v[i,col]*iz
                    i>2 && (correction-=y[i-2,col]/2)
                end
                term=imag(weight*correction)-compensation[i,col]
                updated=total[i,col]+term
                compensation[i,col]=(updated-total[i,col])-term
                total[i,col]=updated
            end
        end
    end
    total
end
apply_transpose(p::ActionPlan,c::AbstractVector{Float64})=vec(apply_transpose(p,reshape(c,:,1)))
end
