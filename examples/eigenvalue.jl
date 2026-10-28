# -D_x^(3/2)u=lambda*u，u(-1)=u(1)=0；用积分逆算子做 Arnoldi。
include("common.jl")

function main()
    n = 500
    solution = MC.eigen_solve(n, :contour; nodes=160)
    solution.converged >= 7 || error("Arnoldi returned fewer than seven converged eigenpairs")
    eigenvalues = inv.(solution.values[1:7])
    println((example=:eigenvalue, n, converged=solution.converged,
             actions=solution.numops))
    for (index, value) in enumerate(eigenvalues)
        println((index, eigenvalue=value))
    end
    (; n, eigenvalues, eigenvectors=solution.vectors[1:7], solution)
end

result = main()
