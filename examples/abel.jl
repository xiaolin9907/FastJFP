# u + 2^(-mu)*lambda*I_x^mu u = f_N，谱右端的单个配置。
include("common.jl")

function main()
    n = 256
    mu = 0.5
    alpha = 0.0
    lambda = 4.0
    rhs = spectral_input(n)
    solver = BandRational.abel_build(n, mu; alpha, lambda, upper=32, nodes=160)
    solution = BandRational.solve!(solver; b=rhs, rtol=1e-14, itmax=400)
    solution.solved || error("Abel coefficient residual did not meet tolerance")
    println((example=:abel, n, mu, alpha, lambda,
             iterations=solution.iterations, residual=solution.residual))
    (; n, mu, alpha, lambda, rhs, solution)
end

result = main()
