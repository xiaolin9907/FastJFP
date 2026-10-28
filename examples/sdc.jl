# u-I_x^(1/2)u = sqrt(pi)*(1+x)/2 - 2*(1+x)^(3/2)/3。
include("common.jl")

function main()
    n = 10
    solution = MC.sdc_solve(n, :contour; nodes=160, rtol=1e-15, maxiter=1000)
    solution.solved || error("SDC discrete residual did not meet tolerance")
    z = cospi.((0:n-1)/(n-1))
    x = 2 .* ((1 .+ z)./2).^2 .- 1
    exact = sqrt(pi).*(1 .+ x)./2
    println((example=:sdc, n, corrections=solution.iterations,
             discrete_residual=last(solution.history).residual,
             maximum_solution_error=norm(solution.u-exact, Inf)))
    (; n, x, values=solution.u, exact, solution)
end

result = main()
