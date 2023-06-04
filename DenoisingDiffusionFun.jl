using Plots, Random, Flux

function make_spiral(n_samples=1000)
    t_min = 1.5π
    t_max = 4.5π

    t = rand(n_samples) * (t_max - t_min) .+ t_min

    x = t .* cos.(t)
    y = t .* sin.(t)

    function normalize_zero_to_one(x)
        x_min, x_max = extrema(x)
        x_norm = (x .- x_min) ./ (x_max - x_min)
        x_norm
    end

    function normalize_neg_one_to_one(x)
        2 * normalize_zero_to_one(x) .- 1
    end

    permutedims([x y], (2, 1)) |> normalize_zero_to_one |> normalize_neg_one_to_one
end

function make_heart(n_sample=1000)
    t = range(0, 2π, length=n_sample)
    scale = 0.1
    x = 16 .* sin.(t) .^ 3 .* scale
    y = (13 .* cos.(t) .- 5 .* cos.(2 .* t) .- 2 .* cos.(3 .* t) .- cos.(4 .* t)) .* scale
    permutedims([x y], (2, 1))
end


using Images, Random, StatsBase, FileIO

img_path = "/Users/katherine/Pictures/Photo Booth Library/Pictures/Photo on 5-31-23 at 10.43 PM.jpg"
img = load(img_path)
A = Gray.(img)

function sample_face(A, n=1000)
    B = canny(A, (Percentile(99), Percentile(98)), 3) #.+ A .* 0.1
    d = ProbabilityWeights((B .|> Float64 |> vec) .^ 2)
    m = sample(CartesianIndices(A), d, n) .|> Tuple .|> collect
    m = (m .- [size(A) |> collect] ./ 2) .* 0.005
    m = [[0 1; -1 0]] .* m
    reduce(hcat, m)
end

mat_scatter(X; kwargs...) = scatter(X[1, :], X[2, :]; kwargs...)
mat_scatter!(X; kwargs...) = scatter!(X[1, :], X[2, :]; kwargs...)

function linear_beta_schedule(num_timesteps, β_start=0.0001f0, β_end=0.02f0)
    LinRange(β_start, β_end, num_timesteps) * 1000 / num_timesteps
end

Xt = make_spiral()
βs = linear_beta_schedule(40, 8e-6, 9e-5)
num_timesteps = 40

βs = linear_beta_schedule(40, 8e-6, 9e-5)
αs = 1 .- βs
α_cumprods = cumprod(αs)
α_cumprod_prevs = [1, (α_cumprods[1:end-1])...]

sqrt_α_cumprods = sqrt.(α_cumprods)
sqrt_one_minus_α_cumprods = sqrt.(1 .- α_cumprods)

function q_sample(x_start, timesteps, noise)
    sqrt_α_cumprods[timesteps]' .* x_start + sqrt_one_minus_α_cumprods[timesteps]' .* noise
end

sqrt_recip_α_cumprods = 1 ./ sqrt.(α_cumprods)
sqrt_recip_α_cumprods_minus_one = sqrt.(1 ./ α_cumprods .- 1)

function predict_start_from_noise(x_t, timesteps, noise)
    sqrt_recip_α_cumprods[timesteps] .* x_t - sqrt_recip_α_cumprods_minus_one[timesteps] .* noise
end

posterior_variance = βs .* (1 .- α_cumprod_prevs) ./ (1 .- α_cumprods)

posterior_mean_coef1 = βs .* sqrt.(α_cumprod_prevs) ./ (1 .- α_cumprods)
posterior_mean_coef2 = (1 .- α_cumprod_prevs) .* sqrt.(αs) ./ (1 .- α_cumprods)

function q_posterior_mean_variance(x_start, x_t, timesteps)
    posterior_mean = posterior_mean_coef1[timesteps] .* x_start + posterior_mean_coef2[timesteps] .* x_t
    (posterior_mean, posterior_variance[timesteps])
end

function p_sample(model, x_t, timesteps; add_noise=true)
    noise = model(x_t, timesteps)
    x_start = predict_start_from_noise(x_t, timesteps, noise)

    posterior_mean, posterior_variance = q_posterior_mean_variance(x_start, x_t, timesteps)
    x_prev = posterior_mean
    if add_noise
        x_prev += sqrt.(posterior_variance) .* randn(size(x_start))
    end
    (x_prev, x_start)
end

###
abstract type AbstractParallel end

_maybe_forward(layer::AbstractParallel, x::AbstractArray, ys::AbstractArray...) = layer(x, ys...)
_maybe_forward(layer::Parallel, x::AbstractArray, ys::AbstractArray...) = layer(x, ys...)
_maybe_forward(layer, x::AbstractArray, ys::AbstractArray...) = layer(x)

struct ConditionalChain{T<:Union{Tuple,NamedTuple}} <: AbstractParallel
    layers::T
end
Flux.@functor ConditionalChain

ConditionalChain(xs...) = ConditionalChain(xs)
function ConditionalChain(; kw...)
    :layers in keys(kw) && throw(ArgumentError("a Chain cannot have a named layer called `layers`"))
    isempty(kw) && return ConditionalChain(())
    ConditionalChain(values(kw))
end

Flux.@forward ConditionalChain.layers Base.getindex, Base.length, Base.first, Base.last,
Base.iterate, Base.lastindex, Base.keys, Base.firstindex

Base.getindex(c::ConditionalChain, i::AbstractArray) = ConditionalChain(c.layers[i]...)

function (c::ConditionalChain)(x, ys...)
    for layer in c.layers
        x = _maybe_forward(layer, x, ys...)
    end
    x
end
###

function p_losses(model, x_start; loss=Flux.mse)
    timesteps = rand(1:40, size(x_start)[2])
    noise = randn(Float32, size(x_start))
    x = q_sample(x_start, timesteps, noise)
    model_out = model(x, timesteps)

    loss(model_out, noise)
end

d_hid = 32
num_timesteps = 40

model = ConditionalChain(
    Parallel(.+, Dense(2, d_hid), Embedding(num_timesteps, d_hid)),
    swish,
    Parallel(.+, Dense(d_hid, d_hid), Embedding(num_timesteps, d_hid)),
    swish,
    Parallel(.+, Dense(d_hid, d_hid), Embedding(num_timesteps, d_hid)),
    swish,
    Dense(d_hid, 2),
)

opt_state = Flux.setup(Adam(0.01), model)

X = make_spiral(10_000_0)
X = make_heart(10_000_0)
X = sample_face(A, 100000)
data = Flux.DataLoader(X; batchsize=32, shuffle=true);


@time for d in data
    dLdM = gradient(p_losses, model, d)[1]
    Flux.update!(opt_state, model, dLdM)
    #p_losses(model, d) |> println
    #p = p_losses(model, d)
    #p > 0.4 ? println(p) : break
end


#r = range(-5, 5, length=50)
#x = [[i, j] for i in r, j in r] |> x -> reduce(hcat, x)
x = randn(2, 10000) #.* 2.5

#mat_scatter(X)
#mat_scatter!(x)

@gif for t in [fill(1, 20)..., [[j for i in 1:2] for j in 1:40]...] |> x->vcat(x...) |> reverse

    global x, x_start = p_sample(model, x, [t])

    p1 = mat_scatter(x_start, lims=(-4, 4), legend=false, title ="X₀")
    p2 = mat_scatter(x, lims=(-4,4), legend=false, title="Xₜ")

    plot(p1, p2)

end

@gif for t in [fill(1, 20)..., [[j for i in 1:2] for j in 1:40]...] |> x->vcat(x...) |> reverse

    global x, x_start = p_sample(model, x, [t])

    p1 = mat_scatter(X[:, 1:1000], lims=(-4, 4), legend=false, title ="Original")
    p2 = mat_scatter(x, lims=(-4,4), legend=false, title="Xₜ")

    plot(p1, p2)

end
