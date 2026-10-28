# epsilon*i^(3/2)*D_x^(3/2)u-x*u=0，u(-1)=0，u(1)=1。
include("common.jl")

function main()
    n = 680
    epsilon = 1e-4
    solver = BandRational.airy_aux_build(n, epsilon; upper=32, nodes=160, max_n=n)
    solution = BandRational.solve!(solver; rtol=1e-13, itmax=200, refinements=3)
    solution.solved || error("Airy coefficient residual did not meet tolerance")
    z = collect(range(-1.0, 1.0; length=2001))
    t = ((1 .+ z)./2).^2
    x = 2 .* t .- 1
    # alpha=1/2：物理解必须包含基底的端点权重 sqrt(t)。
    values = sqrt.(t) .* regular_values(solution.q, z)
    boundary_error = max(abs(first(values)), abs(last(values)-1))
    println((example=:airy, n, epsilon, iterations=solution.iterations,
             residual=solution.residual, boundary_error))
    (; n, epsilon, coefficients=solution.q, x, values, solution)
end

result = main()
