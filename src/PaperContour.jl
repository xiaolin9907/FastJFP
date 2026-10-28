"""Current manuscript entry: fixed M=160, with nodes and weights independent of N."""
module PaperContour
include("JFPContour.jl")
include("GlobalContour.jl")
using .JFPContour: ContourJFP, ReferenceJFP, ContourApplyWorkspace, contour_apply!, solve_abel!, structured_total, structured_matrix
using .GlobalContour: GlobalRule, fixed_plan, build_fixed_abel
export ContourJFP, ReferenceJFP, ContourApplyWorkspace, contour_apply!, solve_abel!, structured_total, structured_matrix
export GlobalRule, action_plan, build_abel, fast_total

action_plan(n,mu;alpha=0.,nodes=160)=fixed_plan(n,GlobalRule(mu;alpha,nodes))
function build_abel(n,mu;alpha=0.,nodes=160,lambda=4.,rtol=1e-14,rhs_drop_tol=1e-100)
    rule=GlobalRule(mu;alpha,nodes)
    build_fixed_abel(n,rule;lambda,rtol,rhs_drop_tol)
end
fast_total(n,mu,b;kwargs...)=solve_abel!(build_abel(n,mu;kwargs...),b)
end
