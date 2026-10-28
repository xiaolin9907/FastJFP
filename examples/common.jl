include(joinpath(@__DIR__, "..", "src", "FastJFP.jl"))
using .FastJFP
using LinearAlgebra, Random

BLAS.set_num_threads(1)

# 原谱输入生成规则：保留机器精度量级的尾部，不丢弃系数。
function spectral_input(n::Int; seed=20260918)
    n >= 3 || throw(ArgumentError("Need at least three coefficients"))
    m = clamp(round(Int, 4n/5), 2, n-1)
    rho = -log(eps(Float64))/(m-1)
    random = 2rand(MersenneTwister(seed), n) .- 1
    factors = copysign.(0.5 .+ 0.5 .* abs.(random), random)
    factors[1] = copysign(1.0, random[1])
    envelope = [j < m ? exp(-rho*(j-1)) : eps(Float64) for j in 1:n]
    factors .* envelope
end

# alpha=0 时，系数可直接在多项式坐标 z 中求值。
regular_values(coefficients, z) = [G.clenshaw(coefficients, x) for x in z]
