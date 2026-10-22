# Batched assay experiment: single body at a random location, unconstrained.
# Planner settings follow run_pomcpow_test.jl, with the paper's 10,000 tree queries.
#
# usage: julia --project=. scripts/run_batch_test.jl POLICY B FIRST LAST
#   POLICY  pomcpow | grid9 | grid16
#   B       holes drilled per batch before assays return (ignored by grids)
#   FIRST, LAST  case indices to run (1-100)

using DelimitedFiles
using POMDPs
using POMCPOW
using Random

using MineralExploration

N_CASES = 100
MAX_BORES = 25
CASE_FILE = "./data/batch_cases.jld"
SAVE_DIR = "./data/batch/"

policy_name = ARGS[1]
B, first_case, last_case = parse.(Int, ARGS[2:4])

# Drill sites on every other cell, as in run_pomcpow_test.jl; grids need every cell, as in test_grid.jl.
grid_spacing = policy_name == "pomcpow" ? 1 : 0
m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=2, grid_spacing=grid_spacing,
                            mainbody_gen=SingleVarNode())
initialize_data!(m, 0)
ds0 = POMDPs.initialstate_distribution(m)

if !isfile(CASE_FILE)
    Random.seed!(0)
    gen_cases(ds0, N_CASES, CASE_FILE)
end
s0s = MineralExploration.load(CASE_FILE)["states"]

up = MEBeliefUpdater(m, 1000, 2.0)
solver = POMCPOWSolver(tree_queries=10000,
                       check_repeat_obs=true,
                       check_repeat_act=true,
                       next_action=NextActionSampler(),
                       k_action=2.0,
                       alpha_action=0.25,
                       k_observation=2.0,
                       alpha_observation=0.1,
                       criterion=POMCPOW.MaxUCB(100.0),
                       final_criterion=POMCPOW.MaxQ(),
                       estimate_value=0.0)
# Julia 1.6 lets the heap grow far past live data during search; collect after each step.
struct Collected{P<:Policy} <: Policy
    policy::P
end
POMDPs.action(p::Collected, b) = (a = POMDPs.action(p.policy, b); GC.gc(); a)

policy = policy_name == "pomcpow" ? POMDPs.solve(solver, m) :
         policy_name == "grid9"   ? GridPolicy(m, 3, 30) :
         policy_name == "grid16"  ? GridPolicy(m, 4, 30) : error("unknown policy $policy_name")
B = policy_name == "pomcpow" ? B : 1
policy = Collected(policy)

mkpath(SAVE_DIR)
tag = "$(policy_name)_B$(B)_$(first_case)-$(last_case)"
summary_file = string(SAVE_DIR, "summary_$tag.csv")
done_cases = isfile(summary_file) ? Int.(readdlm(summary_file, ',')[:, 1]) : Int[]
for i = setdiff(first_case:last_case, done_cases)
    Random.seed!(i) # common random numbers across batch sizes
    s0 = s0s[i]
    b0 = POMDPs.initialize_belief(up, ds0)
    res = run_batch_trial(m, up, policy, s0, b0, B)
    ore = sum(s0.ore_map .>= m.massive_threshold)
    println("case $i  B=$B  ore=$ore  $(res.decision)  holes=$(res.holes)  rounds=$(res.rounds)")
    open(io -> writedlm(io, [i policy_name B ore res.decision res.holes res.rounds res.ret], ','),
         summary_file, "a")
    open(io -> writedlm(io, [[i policy_name B h μ σ] for (h, μ, σ) in res.trace], ','),
         string(SAVE_DIR, "trace_$tag.csv"), "a")
end
