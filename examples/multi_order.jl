# u + sqrt(1+x)*I_x^(1/3)u + I_x^(1/2)[(1+x)^(1/3)u] = f。
include("common.jl")

function main()
    n = 30
    solver = BandRational.linear_build(:variable, n; upper=32, nodes=160)
    solution = BandRational.solve!(solver; rtol=2e-14, itmax=400)
    solution.solved || error("Multi-order coefficient residual did not meet tolerance")
    z = collect(range(-1.0, 1.0; length=2001))
    values = regular_values(solution.q, z)
    exact = MC.exact_values(:variable, z)
    relative_error = norm(values-exact, Inf)/norm(exact, Inf)
    println((example=:multi_order, n, iterations=solution.iterations,
             residual=solution.residual, relative_solution_error=relative_error))
    (; n, coefficients=solution.q, z, values, exact, solution)
end

result = main()
