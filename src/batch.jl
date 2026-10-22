"""
Batched assay feedback.

The planner commits to up to `B` holes before any of their assays return.
While a batch is being filled, each pending hole is assigned its predicted
grade (the "kriging believer" heuristic) so the planner neither re-picks it
nor plans from a stale drill history. A batch closes early if the planner
chooses to stop. MINE/ABANDON is only chosen with every assay in hand.
`B = 1` recovers the original fully sequential setting.
"""
function run_batch_trial(m::MineralExplorationPOMDP, up::MEBeliefUpdater,
                         policy::POMDPs.Policy, s0::MEState, b0::MEBelief, B::Int)
    s, b = s0, b0
    t, ret, holes, rounds, decision = 0, 0.0, 0, 0, :none
    trace = [(0, volume_stats(m, b)...)]
    while !s.decided
        a = POMDPs.action(policy, b)
        if a.type != :drill
            s, b, r = execute(m, up, s, b, a)
            ret += POMDPs.discount(m)^t*r
            t += 1
            decision = a.type
            continue
        end
        batch, bf = [a], b
        while length(batch) < B && length(b.rock_obs) + length(batch) < m.max_bores
            o = MEObservation(predicted_grade(bf, a), false, false)
            bf = POMDPs.update(up, bf, a, o)
            a = POMDPs.action(policy, bf)
            a.type == :drill || break
            push!(batch, a)
        end
        for a in batch
            s, b, r = execute(m, up, s, b, a)
            ret += POMDPs.discount(m)^t*r
            t += 1
        end
        holes += length(batch)
        rounds += 1
        push!(trace, (holes, volume_stats(m, b)...))
    end
    return (ret=ret, holes=holes, rounds=rounds, decision=decision, trace=trace)
end

function execute(m::MineralExplorationPOMDP, up::MEBeliefUpdater, s, b, a)
    sp, o, r = POMDPs.gen(m, s, a, up.rng)
    return sp, POMDPs.update(up, b, a, o), r
end

predicted_grade(b::MEBelief, a::MEAction) =
    mean(p.ore_map[a.coords[1], a.coords[2], 1] for p in b.particles)

function volume_stats(m::MineralExplorationPOMDP, b::MEBelief)
    vols = [sum(p.ore_map .>= m.massive_threshold) for p in b.particles]
    return mean(vols), std(vols)
end
