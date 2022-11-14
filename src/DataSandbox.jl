module DataSandbox

using TOML

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)
using Distributions
using Random

using Printf
using BSON

include("data.jl")
include("simulate.jl")

export generate_system_data
export build_function, simulate

end
