using SciMLBase, ModelingToolkit, GalacticOptim, NeuralPDE, Flux, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum
using IntervalSets
using Plots
using ShowCode
using Cthulhu


ty(v) = typeof(v)
fn(ty) = fieldnames(ty)
fnty(v) = fn(ty(v))
tyfnty(v) = map(f->println(string((f, ty(getfield(v, f))))), fnty(v))

@parameters x1, x2, x3, x4
@variables u(..)
D1 = Differential(x1)
D2 = Differential(x2)
D3 = Differential(x3)
D4 = Differential(x4)

# System of pde
eqs = [D1(D1(u(x1, x2, x3, x4))) +
       D2(D2(u(x1, x2, x3, x4))) +
       D3(D3(u(x1, x2, x3, x4))) +
       D4(D4(u(x1, x2, x3, x4))) ~ 0,
      ]

# Initial and boundary conditions
bcs = [
    u(x1,0,0,0) ~ 1*x1,
    u(0,x2,0,0) ~ 2*x2,
    u(0,0,x3,0) ~ 3*x3,
    u(0,0,0,x4) ~ 4*x4,
    ]

# Space and time domains
domains = [
    x1 ∈ Interval(0.0,1.0),
    x2 ∈ Interval(0.0,1.0),
    x3 ∈ Interval(0.0,1.0),
    x4 ∈ Interval(0.0,1.0),
    ]
@named pde_system = PDESystem(eqs,bcs,domains,[x1,x2,x3,x4],[u(x1,x2,x3,x4)])


# Neural network
chain = [FastChain(FastDense(4,15,Flux.tanh),FastDense(15,1))]

stochastic_strategy = NeuralPDE.StochasticTraining(128)
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

discretization = NeuralPDE.PhysicsInformedNN(chain,stochastic_strategy; init_params = initθ)


prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

iteration = [1]
function cb(p, l)
    if iteration[1] % 10 == 0
        @info "Iteration: ", iteration[1]
    end
    iteration[1] += 1
    return false
end
res = GalacticOptim.solve(prob,BFGS(); maxiters=1000, cb=cb)
phi = discretization.phi

sol_func2(x) = phi[1](x, res.u)[1]

function gen_range_slices(vardomains::AbstractVector{Symbolics.VarDomainPairing}, num_points_per_dim=128)
    map(vardomain -> range(vardomain.domain, num_points_per_dim), vardomains)
end

num_points_per_dim = 4


function eval_func_slice(func, vardomains::AbstractVector{Symbolics.VarDomainPairing}, symbols::AbstractVector{Num}, num_points_per_dim)
    sym_indices = map(sym->findmax(map(vardomain->isequal(vardomain.variables, sym.val), vardomains))[2], symbols)
    
    linear_indices = axes(ones(repeat([num_points_per_dim], 4)...))
    cart_indices = CartesianIndices(linear_indices)
    for I in eachindex(cart_indices)
        @show I
    end

    range_slices = gen_range_slices(vardomains, num_points_per_dim)
    @show range_slices

    value_from_index(index) = []

end

function gen_subslices(constant_values, nonconstant_ranges)
    # 
end



slice_vars = [x2, x1, x3]
func_sliced = eval_func_slice(sol_func2, domains, slice_vars, num_points_per_dim)

@generated function eval_pde_func(f, x::Union{Real, AbstractVector{<:Real}}...) 
    broadcast_dims = map(x_i->x_i <: AbstractVector, x)
    broadcast_indices = Set(map(first, filter(i_b->i_b[2], collect(enumerate(broadcast_dims)))))
    N = length(x)
    eltypes = map(x_i -> x_i <: AbstractVector ? eltype(x_i) : x_i, x)
    promotion_type = promote_type(eltypes...)
    quote
        # need to make a 2D array with the cartesian product of all the broadcasted 
        broadcast_lengths = length.(x)
        num_points = prod(broadcast_lengths)
        point_array = Array{$promotion_type, 2}(undef, $N, num_points) # TODO: make the AbstractArray type be more general
        iter_indices = CartesianIndices(tuple(broadcast_lengths...))
        for (i, index) in enumerate(iter_indices)
            for j in 1:$N
                if j in $(broadcast_indices)
                    point_array[j, i] = x[j][index.I[j]]
                else
                    point_array[j, i] = x[j]
                end
            end
        end

        # apply the function to the point array
        output_pre_reshape = f(point_array)
        out_dim = size(output_pre_reshape, 1)
        output = reshape(output_pre_reshape, (out_dim, broadcast_lengths...))
        return (N=$N, broadcast_dims=$(broadcast_dims), broadcast_indices=$(broadcast_indices), broadcast_lengths=broadcast_lengths, 
            num_points=num_points, point_array=point_array, shape=size(point_array), iter_indices=iter_indices,
            output_pre_reshape=output_pre_reshape, out_dim=out_dim, output=output,
            eltypes=$eltypes, promotion_type=$promotion_type)
    end

end

@generated function sample_grid(x::Union{<:Real, AbstractVector{<:Real}}...; flat=false) 
    broadcast_dims = map(x_i->x_i <: AbstractVector, x)
    broadcast_indices = Set(map(first, filter(i_b->i_b[2], collect(enumerate(broadcast_dims)))))
    N = length(x)
    eltypes = map(x_i -> x_i <: AbstractVector ? eltype(x_i) : x_i, x)
    promotion_type = promote_type(eltypes...)
    quote
        # need to make a 2D array with the cartesian product of all the broadcasted 
        broadcast_lengths = length.(x)
        num_points = prod(broadcast_lengths)
        iter_indices = CartesianIndices(tuple(broadcast_lengths...))

        if $flat
            point_array = Array{$promotion_type, 2}(undef, $N, num_points) # TODO: make the AbstractArray type be more general
            for (i, index) in enumerate(iter_indices)
                for j in 1:$N
                    if j in $(broadcast_indices)
                        point_array[j, i] = x[j][index.I[j]]
                    else
                        point_array[j, i] = x[j]
                    end
                end
            end
        else
            point_array = Array{$promotion_type, ($N + 1)}(undef, $N, broadcast_lengths...) # TODO: make the AbstractArray type be more general
            for (i, index) in enumerate(iter_indices)
                for j in 1:$N
                    if j in $(broadcast_indices)
                        point_array[j, index] = x[j][index.I[j]]
                    else
                        point_array[j, index] = x[j]
                    end
                end
            end
        end

        return (N=$N, broadcast_dims=$(broadcast_dims), broadcast_indices=$(broadcast_indices), broadcast_lengths=broadcast_lengths, 
            num_points=num_points, point_array=point_array, shape=size(point_array), iter_indices=iter_indices)
    end

end

struct MultiDimensionalFunction{FuncType, IVsType, DVsType, IVDVsType}
    f::FuncType
    ivs::IVsType
    dvs::DVsType
    ivs_in_dvs::IVDVsType
end

x1s = 0:0.1:1
x1i = 0.1
x2s = 0:0.2:1
x2i = 0.4
ys = 0:0.1:0.2
f_eval(x) = phi[1](x, res.u)
allys = eval_pde_func(f_eval, ys, ys, ys, Float32(x1i))
eltypes = allys.eltypes
promote_type(eltypes...)
#eval_pde_func(x1i, x2i)
#ful2 = eval_pde_func(Float32.(x1s), x2s)
#ful3 = eval_pde_func(Float32.(x1s), x2s, x2s)
#eval_pde_func(x1i, x2s)
#bd = eval_pde_func(x1i)
#bd = eval_pde_func(x1s)

lengths2 =  (length).([x1s, x2s])
indices = CartesianIndices(tuple(lengths2...))


"""
things I want to plot:

pde values and slices of them (heterogenous)



"""


"""
desired interface:

AbstractPDESolution <: AbstractNoTimeSolution
FunctionPDESolution <: AbstractPDESolution
PointwisePDESolution <: AbstractPDESolution

would like to treat both as a function and both as a grid

res::FunctionPDESolution = solve(prob, opt)
sample_slice()


"""



"""
const TYPE_LENGHT_LIMIT = Ref(20)
function Base.print_type_stacktrace(io, type; color=:normal)
str = first(sprint(show, type, context=io), TYPE_LENGHT_LIMIT[])
i = findfirst('{', str)
if isnothing(i) || !get(io, :backtrace, false)::Bool
printstyled(io, str; color=color)
else
printstyled(io, str[1:prevind(str,i)]; color=color)
printstyled(io, str[i:end]; color=:light_black)
end
end """