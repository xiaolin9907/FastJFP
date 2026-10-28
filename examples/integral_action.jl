# 计算 I_x^(1/2)(1+x)，不求解方程，也不测量耗时。
include("common.jl")

function main()
    n = 128
    mu = beta = 0.5
    workspace = G.IntegralWorkspace(n, mu, beta; alpha=0.0, nodes=160)
    input = zeros(n)
    input[1:3] .= G.polyv(2, 2.0)  # 1+x=2*((1+z)/2)^2
    output = zeros(n+1)
    G.integral!(output, workspace, input)

    z = collect(range(-1.0, 1.0; length=201))
    x = 2 .* ((1 .+ z)./2).^2 .- 1
    values = regular_values(output, z)
    exact = 4/(3sqrt(pi)) .* (1 .+ x).^1.5
    println((example=:integral_action, n, maximum_error=norm(values-exact, Inf)))
    (; n, coefficients=output, x, values, exact)
end

result = main()
