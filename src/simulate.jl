# Functions to facilitate simple simulation of a system

function simulate(f, x0, strategy, steps; dt=0.1, label_fcn=nothing, stop_label=nothing)

    state_history = zeros(length(x0), steps)
    state_history[:,1] = x0
    t = 0.
    initial_label = isnothing(label_fcn) ? nothing : label_fcn(x0)

    for i=2:steps
            if isnothing(strategy)
                state_history[:,i] = f(state_history[:,i-1], t)
            else
                state_history[:,i] = f(state_history[:,i-1], strategy(state_history[:,i-1], t))
            end

            if !isnothing(label_fcn)
                    if isnothing(stop_label)
                            if label_fcn(state_history[:,i]) != initial_label
                                    state_history = state_history[:, 1:i]
                                    break
                            end
                    else
                            if label_fcn(state_history[:,i]) == stop_label
                                    state_history = state_history[:, 1:i]
                                    break
                            end
                    end
            end
            t += dt
    end

    return state_history
end

function simulate(f, x0, steps; dt=0.1, label_fcn=nothing, stop_label=nothing)
   return simulate(f, x0, nothing, steps, dt=dt, label_fcn=label_fcn, stop_label=stop_label) 
end

function build_function(config_filename::String; full_observability=true, random_seed=11)   # TODO: use measurement noise someday
    config = parse_TOML_file(config_filename)
    mt = MersenneTwister(random_seed)

    f_ex = Meta.parse(config["system_function"])
    dyn_fcn = @RuntimeGeneratedFunction(f_ex)

    noise_config = config["noise"]
    n_dims_out = length(config["data"]["sample_range"][1]) # TODO: Assumes same dims in input and output

    process_noise_dist = nothing
    if noise_config["process_distribution"] == "Gaussian"
        process_distribution = Normal(noise_config["process_mean"], noise_config["process_std"])
        return (x,t) -> dyn_fcn(x) + rand(mt, process_distribution, (n_dims_out, 1)) 
    else
        return (x,t) -> dyn_fcn(x)
    end
end