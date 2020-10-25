### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 0fdb6d38-16da-11eb-16c9-75999a705c48
#=
MLJ: Generic machine learning framework
DataFrames: Like pandas
CSV: Can load csv files crazy fast
=#
using MLJ, DataFrames, CSV

# ╔═╡ 194b7b14-1703-11eb-3ca1-79311462911b
using D3Trees

# ╔═╡ 3cefce80-1703-11eb-1756-b5a86962fd0f
md"# Classification 🌳 Demo"

# ╔═╡ de3e40fa-1703-11eb-3587-7bd3ddd57fd7
md"## Load and view data"

# ╔═╡ 23510310-16e3-11eb-1d1f-cf1e0ff937c0
df_music = CSV.read("music.csv");

# ╔═╡ 4cd344c8-16e3-11eb-33f7-1dfe56648f20
head(df_music)

# ╔═╡ 546102de-16e3-11eb-0f92-fd3f95d82f49
describe(df_music)

# ╔═╡ 0e5d7166-1704-11eb-3296-ad2292011bb6
md"## Model"

# ╔═╡ fb8dcc72-16e3-11eb-2f82-1bdb0f71c505
y, X = unpack(df_music, ==(:genre), c -> true; :genre => Multiclass);

# ╔═╡ 26f66bee-16e4-11eb-37b8-35c24756cf6c
tree_model = @load DecisionTreeClassifier verbosity=0

# ╔═╡ b9cadbd0-16e4-11eb-2dd4-e53d149c7a93
tree = machine(tree_model, X, y)

# ╔═╡ 85e6eb6e-16f4-11eb-012c-1d301ed9e29b
evaluate!(
	tree,
	resampling = Holdout(fraction_train=0.8, shuffle=true),
	measure = accuracy,
	operation = predict_mode
)[2]

# ╔═╡ 6f7f1300-1704-11eb-38d4-4d9e6e3e1687
md"## Visualize"

# ╔═╡ 7b7962d2-1704-11eb-1f2e-bfd3e34a9ee8
md"## Helper functions"

# ╔═╡ 0a807d4a-16fc-11eb-0e68-b5a90391bd0a
begin
	## directly copied from StatsBase.jl to avoid the dependency ##

	const RealArray{T<:Real,N} = AbstractArray{T,N}
	const IntegerArray{T<:Integer,N} = AbstractArray{T,N}


	function _check_randparams(rks, x, p)
		n = length(rks)
		length(x) == length(p) == n || raise_dimerror()
		return n
	end

	# Ordinal ranking ("1234 ranking") -- use the literal order resulted from sort
	function ordinalrank!(rks::AbstractArray, x::AbstractArray, p::IntegerArray)
		n = _check_randparams(rks, x, p)

		if n > 0
			i = 1
			while i <= n
				rks[p[i]] = i
				i += 1
			end
		end

		return rks
	end


	"""
		ordinalrank(x; lt = isless, rev::Bool = false)

	Return the [ordinal ranking](https://en.wikipedia.org/wiki/Ranking#Ordinal_ranking_.28.221234.22_ranking.29)
	("1234" ranking) of an array. The `lt` keyword allows providing a custom "less
	than" function; use `rev=true` to reverse the sorting order.
	All items in `x` are given distinct, successive ranks based on their
	position in `sort(x; lt = lt, rev = rev)`.
	Missing values are assigned rank `missing`.
	"""
	ordinalrank(x::AbstractArray; lt = isless, rev::Bool = false) =
		ordinalrank!(Array{Int}(undef, size(x)), x, sortperm(x; lt = lt, rev = rev))

end

# ╔═╡ 4487abd8-16fc-11eb-1a50-17e64a97c1a8
begin
	import MLJModels.DecisionTree: Node, Leaf
	
	function name(node::Node)
		featval = string(isa(node.featval, Real) && !isa(node.featval, Bool) ? round(node.featval; digits=2) : node.featval)
		"Feature: $(string(node.featid))\nThreshold: $(featval)"
	end

	function name(leaf::Leaf)
		matches = findall(leaf.values .== leaf.majority)
		ratio = string(length(matches)) * "/" * string(length(leaf.values))
		majority = string(isa(leaf.majority, Real) && !isa(leaf.majority, Bool) ? round(leaf.majority; digits=2) : leaf.majority)
		"Leaf\n$(majority): $(ratio)"
	end

	function add_children!(model::Leaf, tree::Vector{Vector{Int}}, names::Vector{String}, ordering=Vector{Int}, indent_counter=1)
		push!(tree, Int[])
	end

	function add_children!(model::Node, tree::Vector{Vector{Int}}, names::Vector{String}, ordering=Vector{Int}, indent_counter=1)
		push!(tree, [1,2]) # placeholders for correct indices

		# left
		push!(names, name(model.left))
		push!(ordering, indent_counter)
		add_children!(model.left, tree, names, ordering, indent_counter+1)

		# right
		push!(names, name(model.right))
		push!(ordering, indent_counter)
		add_children!(model.right, tree, names, ordering, indent_counter+1)
	end

	"""
		D3DecisionTree(model)

	Construct an interactive tree visualization from a `DecisionTree.jl` model
	to be displayed using D3 in a browser or Jupyter notebook.
	"""
	function D3DecisionTree(model::Node)
		modeltree = Vector{Vector{Int}}()
		names = [name(model)]::Vector{String}
		ordering = Int[0]

		add_children!(model, modeltree, names, ordering)

		ordering = ordinalrank(ordering)
		n = length(ordering)

		modeltreeout = Vector{Vector{Int}}(undef, n)
		namesout = Vector{String}(undef, n)

		for i=1:n
			o = ordering[i]
			modeltreeout[o] = modeltree[i]
			namesout[o] = names[i]
		end

		# fill in correct indices
		c=2
		for i=1:n
			if modeltreeout[i] != Int[]
				modeltreeout[i] = [c, c+1]
				c += 2
			end
		end

		D3Tree(modeltreeout, text=namesout, init_expand=10)
	end
end

# ╔═╡ a888ee66-16fd-11eb-276f-3bd794070a34
D3DecisionTree(fitted_params(tree).tree)

# ╔═╡ c7653b2e-16f7-11eb-3535-a95fb7d84d42
note(text) = Markdown.MD(Markdown.Admonition("note", "Note", [text]))

# ╔═╡ 4281fab4-16f7-11eb-36ad-697541b76d06
note(md"""
This just does the following in one step:
```julia
train, test = partition(eachindex(y), .8, shuffle=true)
fit!(tree, rows=train)
ȳ = predict_mode(tree, rows=test)
accuracy(y[test], ȳ)
```
""")

# ╔═╡ Cell order:
# ╟─3cefce80-1703-11eb-1756-b5a86962fd0f
# ╠═0fdb6d38-16da-11eb-16c9-75999a705c48
# ╟─de3e40fa-1703-11eb-3587-7bd3ddd57fd7
# ╠═23510310-16e3-11eb-1d1f-cf1e0ff937c0
# ╠═4cd344c8-16e3-11eb-33f7-1dfe56648f20
# ╠═546102de-16e3-11eb-0f92-fd3f95d82f49
# ╟─0e5d7166-1704-11eb-3296-ad2292011bb6
# ╠═fb8dcc72-16e3-11eb-2f82-1bdb0f71c505
# ╠═26f66bee-16e4-11eb-37b8-35c24756cf6c
# ╠═b9cadbd0-16e4-11eb-2dd4-e53d149c7a93
# ╠═85e6eb6e-16f4-11eb-012c-1d301ed9e29b
# ╟─4281fab4-16f7-11eb-36ad-697541b76d06
# ╟─6f7f1300-1704-11eb-38d4-4d9e6e3e1687
# ╠═194b7b14-1703-11eb-3ca1-79311462911b
# ╠═a888ee66-16fd-11eb-276f-3bd794070a34
# ╟─7b7962d2-1704-11eb-1f2e-bfd3e34a9ee8
# ╟─0a807d4a-16fc-11eb-0e68-b5a90391bd0a
# ╟─4487abd8-16fc-11eb-1a50-17e64a97c1a8
# ╟─c7653b2e-16f7-11eb-3535-a95fb7d84d42
