# Standalone reusable contour action; extracted without PDE or ADI code.

struct ContourApplyWorkspace
    plan::ContourJFP.ActionPlan
    rhs::Matrix{Float64}
    ac::Matrix{Float64}
    total::Matrix{Float64}
    compensation::Matrix{Float64}
    solution::Matrix{ComplexF64}
    inverse_diagonal::Matrix{ComplexF64}
    upper2::Matrix{ComplexF64}
end

function ContourApplyWorkspace(p::ContourJFP.ActionPlan,b::Int)
    n=p.n
    inverse=Matrix{ComplexF64}(undef,n,p.nodes); upper2=similar(inverse)
    for k in eachindex(p.shifts),i in 1:n
        z=p.shifts[k]
        inverse[i,k]=inv(p.mu*(i-1)/2-z*(i==1 ? 1.0 : 0.5))
        upper2[i,k]=(p.mu*(i+1)+z)/2
    end
    ContourApplyWorkspace(p,zeros(n,b),zeros(n,b),zeros(n,b),zeros(n,b),
        zeros(ComplexF64,n,b),inverse,upper2)
end

"""Same finite contour action as ContourJFP.apply, with reusable buffers.
The n-by-M tables contain only diagonals of the individual three-band
Euler resolvents, never columns or factors of a time integral matrix.
"""
function contour_apply!(out::AbstractMatrix{Float64},w::ContourApplyWorkspace,C::AbstractMatrix{Float64})
    p=w.plan; n,b=size(C)
    size(C)==size(w.rhs)==size(out) || throw(DimensionMismatch())
    rhs=w.rhs;ac=w.ac;total=w.total;compensation=w.compensation;solution=w.solution
    @inbounds for col in 1:b,i in 1:n
        rhs[i,col]=(i==1 ? 1.0 : 0.5)*C[i,col]
        i+2<=n && (rhs[i,col]-=C[i+2,col]/2)
        ac[i,col]=p.mu*(i-1)/2*C[i,col]
        i+1<=n && (ac[i,col]+=p.mu*i*C[i+1,col])
        i+2<=n && (ac[i,col]+=p.mu*(i+1)/2*C[i+2,col])
        total[i,col]=p.constant*C[i,col]
        compensation[i,col]=0.0
    end
    @inbounds for k in eachindex(p.shifts)
        z=p.shifts[k];weight=p.weights[k];factored=abs(z)>=p.threshold;iz=inv(z)
        for col in 1:b
            for i in n:-1:1
                value=factored ? ac[i,col]*iz : complex(rhs[i,col])
                i+1<=n && (value-=p.mu*i*solution[i+1,col])
                i+2<=n && (value-=w.upper2[i,k]*solution[i+2,col])
                solution[i,col]=value*w.inverse_diagonal[i,k]
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
    ContourJFP.pole_correction!(total,p,rhs)
    fill!(out,0.0)
    @inbounds for col in 1:b,i in 1:n
        out[i,col]+=total[i,col]/2
        i>1 && (out[i-1,col]+=total[i,col]/4)
        i<n && (out[i+1,col]+=(i==1 ? 0.5 : 0.25)*total[i,col])
    end
    out
end

struct InplaceFunctionOperator{F}
    n::Int
    f!::F
end
Base.size(A::InplaceFunctionOperator)=(A.n,A.n)
Base.size(A::InplaceFunctionOperator,d::Int)=d<=2 ? A.n : 1
Base.eltype(::InplaceFunctionOperator)=Float64
LinearAlgebra.mul!(y::AbstractVector,A::InplaceFunctionOperator,x::AbstractVector)=A.f!(y,x)

