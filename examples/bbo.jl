# u + I_x^(1/2)u + I_x^1 u = -1；重构 v=1+I_x^1 u。
include("common.jl")

function main()
    n = 50
    solver = BandRational.linear_build(:bbo, n; upper=32, nodes=160)
    solution = BandRational.solve!(solver; rtol=2e-14, itmax=400)
    solution.solved || error("BBO coefficient residual did not meet tolerance")
    coefficients = zeros(n+8)
    G.integral!(coefficients, solver.full.post, solution.u)
    coefficients[1] += 1
    z = collect(range(-1.0, 1.0; length=2001))
    values = regular_values(coefficients, z)
    exact = MC.exact_values(:bbo, z)
    relative_error = norm(values-exact, Inf)/norm(exact, Inf)
    println((example=:bbo, n, iterations=solution.iterations,
             residual=solution.residual, relative_solution_error=relative_error))
    (; n, coefficients, z, values, exact, solution)
end

result = main()
