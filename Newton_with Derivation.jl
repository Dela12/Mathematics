### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ 621b6832-de86-11ec-1ee8-3d4c78a7da67
begin 
    ENV["LANG"]="C"
    using PlutoUI
    using PyPlot
    using LinearAlgebra
    using ForwardDiff
	 using ForwardDiff: jacobian
    using DiffResults
	using LaTeXStrings
    PyPlot.svg(true)
	function plothistory(history::Vector{<:Number})
		clf()
		semilogy(history)
		xlabel("steps")
		ylabel("residual")
		grid()
        gcf()
    end;
end;

# ╔═╡ 5ec2097a-89ac-45f4-85e9-b02cf16283e0
using DataFrames

# ╔═╡ 061e7727-fb64-42a9-9555-acc7c2a6e9d5
md"""
 #### Numerical Example of Newton's method in Multidimension

We shall discuss how to solve Nonlinear Systems of Equations using Newton's method. We will demonstrate how to compute the Jacobian using automatic differentiation and further show quadratic convergence of Newton's method if any exist for our problem. All these shall be done in Julia.jl. 

###### Basic Idea

This method is basically based on the idea of linear approximation. The tangent line at a given point is a good approximation of the function. Newton's iteration approximates a function based on its tangent line, then takes the tangent line's zero as the next guess $$\phi_i$$.

We begin with an initial (suitable) guess $$\phi_0$$. Then, $$\phi_0$$ is used in to produce $$\phi_1$$ and $$\phi_1$$ is further used to produced $$\phi_2$$ and so on. The objective is that one will ultimately find a value $$\phi_i$$ that is close enough to the desired solution.

Recall the Taylor Series approximation of polynomials of increasing powers;

$$f(\phi) = f(\phi_0) + (\phi - \phi_0)f'(\phi_0) + \frac{1}{2}(\phi-\phi_0)^2f''(\phi_0)+...$$

where $$f'$$ and $$f''$$ are the first and second derivative of $$f$$ at $$\phi_0$$. If we consider the first two terms of the Taylor series expansion we have:

$$f(\phi) = f(\phi)+f'(\phi_0)(\phi-\phi_0)$$

setting $$f(\phi) = 0$$,

$$0 = f(\phi_0) + (\phi - \phi_0)f'(\phi_0)$$

$$(\phi - \phi_0)f'(\phi_0) = -f(\phi_0)$$

$$(\phi - \phi_0) = \frac{-f(\phi_0)}{f'(\phi_0)}$$

$$(\phi - \phi_0) = \frac{-f(\phi_0)}{f'(\phi_0)}$$

In general, Newton's iteration in 1D is given by

$$\phi_{i+1}= \phi_i - \frac{f(\phi_i)}{f'(\phi_i)}$$

###### In multidimension:

We extend Newton's iteration in 1D into multi-dimensional system. 

Essentially, we are finding the zeroes of $$F(ϕ) = b$$

where, $$F(\phi) := A\phi + \Psi(\phi)$$ and $$b = 0.$$

In these case, $$F(\phi)$$ is regarded as a nonlinear operator $$F: D \longrightarrow \mathbb{R}^n$$ where $$D \subset \mathbb{R}^n$$ is the domain of definition, $$\phi = [\phi_1,\phi_2,...,\phi_n]^T \in \mathbb{R}^n$$ and $$b = [0,0,...,0]^T \in \mathbb{R}^n$$ is a zero vector. 

The scalar $$\phi_i$$ in the 1D case is replaced by vector $$\phi_i$$ and we left multiply $$F(\phi_i)$$ by the inverse of it's Jacobian matrix $$F'(\phi_i) = J(\phi_i).$$ 

This results in a multidimensional Newton's iteration equation: 

$$\phi_{i+1} = \phi_i - J(\phi_i)^{-1}F(\phi_i)$$

The inverse of $F(\phi)$ is expensive to compute. Hence instead of computing $J(\phi)^{-1}$, we solve a linear equation by rearranging the iteration eqaution  

$$J(\phi_i)\delta_i = - F(\phi_i)$$

where $$\delta_i = \phi_{i+1} - \phi_i$$ using an initial guess $$\phi_0\in \mathbb{R}^n$$ and $$\delta_i \in \mathbb{R}^n$$.

we can then find $$\delta_i$$ such that,

$$\delta_i = -J(\phi_i)^{-1}F(\phi_i)$$. 

Hence we can replace $$-J(\phi_i)^{-1}F(\phi_i)$$ with $$\delta_i$$ in the iteration equation such that

$$\phi_ {i+1} = \phi_{i} - \delta_i$$

"""

# ╔═╡ 4644cb13-bc49-4a6a-aace-78057353d298
md"""
#### Problem Setup

Consider the following nonlinear algebraic system which we assume to be from the finite volume discretization of the Poisson-Boltzmann and Poission-Bikermann models:

$A_k\phi + \Psi_k(\phi) = b_k$

where, $F(\phi) := A_k\phi + \Psi_k(\phi)$ such that 

$F(\phi) = b_k.$

We define the components of $F(\phi)$ and $b_k$ as follows:

$$F(\phi):=\begin{pmatrix}
  3\phi_1 - cos(\phi_2\phi_3) - \frac{1}{2} \\
  \phi_1^{2} - 81(\phi_2 + 0.1)^2 + sin (\phi_3) + 1.06 \\
  e^{-\phi_1 \phi_2} + 20\phi_3 + \frac{10\pi - 3}{3} \\
  \end{pmatrix}= \begin{pmatrix}
  0\\
  0\\
  0
  \end{pmatrix}$$

The system is a nonlinear system with three equations and three unknowns. To solve this system, we shall implement the following.

"""

# ╔═╡ c2ccf8e2-235b-4575-adcc-963f76adbf47
md"""
#### Implementation

I tried solving the above equation for different initial conditions, for some, it converges in at least 9 iterations while for some other initial conditions it diverges badly. The measure of convergence is checked by computing the norm $$||\phi_{i+1} - \phi_i||$$ in every iteration step.
"""

# ╔═╡ 58c8d6db-5acd-484e-87ee-8b0a3b87e94d
md"""
###### Step one : Inital Vector

The first step of Newton's method is to determine an initial vector. A standard way of determining an initial vector will be to compute the intersection of all three function. As in "reference", let$\phi_0$ be our initial vector,  

$\begin{align*} \phi_0 = 
\begin{bmatrix}
0.1 & 0.1 & -0.1
\end{bmatrix}^T
\end{align*}$ 

and let $\phi^{*}$ be the solution to the system $F(\phi) = b$.
"""

# ╔═╡ c44bc767-537f-455f-9344-3b43c1e7b36f
ϕ₀ = [0.1,0.1,-0.1]

# ╔═╡ 031674ed-4ef7-42a8-b884-ca5ce143bd89
md"""
###### Step Two : Define the System

Next, we define the components of the nonlinear system $F(\phi) = b$. We define first the left hand vector $$F(\phi)$$ followed by the right hand vector $$b$$.
"""

# ╔═╡ 80074f16-0e6a-4981-b11a-708c38320b3d
function G(ϕ)
	[3*ϕ[1] - cos(ϕ[2]*ϕ[3])- 1/2;
	ϕ[1]^2 - 81*(ϕ[2] + 0.1)^2 + sin(ϕ[3]) + 1.06;
	exp(-ϕ[1]* ϕ[2]) + 20*ϕ[3] + (10*pi - 3)/3]
end

# ╔═╡ d8681c01-a464-46ba-bcd4-a14794492082
F(ϕ) = G(ϕ)

# ╔═╡ ffe7267f-3739-417a-8af5-4604289178eb
b = [0,0,0]

# ╔═╡ 3e32e664-0fc6-46cf-9164-b0f11e351d53
md"""
###### Step Three : Evalaute $F$ at $\phi_0$
Now, we evaluate $$F(\phi)$$ at the inital vector $$\phi$$.
"""

# ╔═╡ ec51b7e2-302a-4fe1-bc01-7bf6996a50a6
F(ϕ₀)

# ╔═╡ 9679734f-1d76-44f3-8e0a-e365869192b3
md"""
After evaluating $F(\phi_0)$, we obtained a row vector. By transposing the row vector, we have the following column vector:

$\begin{align*} F(\phi_0) = 
\begin{bmatrix}
 -1.19995 \\ -2.26983 \\ 8.46203
\end{bmatrix}
\end{align*}$
"""

# ╔═╡ ef5cfe9f-8462-4a26-b95c-2b3bbe8e07ee
md"""
###### Step four : Residual

Here, the goal isto obtain the residual needed for Newton's iteration. We compute the residual at the inital vector $$\phi_0$$. 
"""

# ╔═╡ cba9bf8d-ea27-484c-9e58-1f424ff04209
begin
	r(ϕᵢ) = F(ϕᵢ) - b    #i = 0,...n
	r(ϕ₀)
end

# ╔═╡ 5788401c-a830-4096-b239-b44a571a459f
md"""
We transpose the compute the residual above and obtain the following column vector

$\begin{align*} r(\phi_0) = 
\begin{bmatrix}
 -1.19995 \\ -2.26983 \\ 8.46203
\end{bmatrix}
\end{align*}$
"""

# ╔═╡ e2eb4ecd-2233-4be9-884a-fd0f91516391
md"""
###### Step five : Jacobian

We compute and evaluate the Jacobian of $F(\phi)$ at $\phi_0$. We denote the Jacobian at $$\phi_0$$ by $J(\phi_0)$.

This is done using Julia's automatic differetiation package "ForwardDiff: jacobian". Reference is herein made to the Scientific Computing Lecture by Dr. Jurgen Fuhrmann(WS21/22) for coding idea.

To begin with, we create a result buffer for our $3 \times 3$ nonlinear system in order to use automatic differentition to compute the Jacobian. We use the DiffResults.JacobianResult package. This is saved as d_results.

Next, we call the ForwardDiff.jacobian! package on $F$ with the inital vector $\phi_0$ as input data. This mutates the d_results. Additionally, within the d_results, we otain the value for the differential results. Finally, we obtain the Jacobian at $$\phi_0$$, which is a $3 \times 3$ matrix as well as it's inverse $$J^{-1}(\phi_0).$$
"""

# ╔═╡ 5d9a24d1-246f-43c0-87ec-30090f2b4c4b
d_result=DiffResults.JacobianResult(ϕ₀)

# ╔═╡ 6057daab-03ba-4a97-9799-3af305854386
ForwardDiff.jacobian!(d_result,F,[0.1, 0.1, 0.1])

# ╔═╡ 4e2f13aa-6cc9-44e1-8052-517042489c7b
DiffResults.value(d_result)

# ╔═╡ cd3ab5fb-6186-40e6-8731-bb814b4286fe
J(ϕ₀) = DiffResults.jacobian(d_result)

# ╔═╡ fb49df9b-765d-4a77-8da2-b97c0aed80b5
DiffResults.jacobian(d_result)

# ╔═╡ 89925314-040d-4dfe-9489-c923a929ab98
inv(J(ϕ₀))

# ╔═╡ 7d108cb8-0f67-42f3-bb2e-eb580302d8cb
md"""
The results are written explicitly below, where the Jacobian is given by:

$\begin{align*} J(\phi_0) = \begin{bmatrix}
3.0 &  0.000999983 & 0.000999983 \\
0.2 & -32.4 & 0.995004 \\
-0.099005 & -0.099005 & 20.0 \\
\end{bmatrix}.
\end{align*}$

and the inverse of the Jacobian is given as,

$\begin{align*} J(\phi_0)^{-1} = \begin{bmatrix}
0.333332 &  1.03404e-5 & -1.71808e-5 \\
0.0021086 & -0.0308688 & 0.00153563 \\
0.00166051 & -0.000152757 &  0.0500075 \\
\end{bmatrix}.
\end{align*}$
"""

# ╔═╡ b53d1663-eb65-4ef8-93fc-d9f619548ce0
md"""

###### Step Six : linear system and solution update
Next we shall solve a linear system for $\delta \in \mathbb{R}^n$ : 

$\begin{align*} 
J(\phi_0)\delta_0 = r(\phi_0)
\end{align*}$ 

After computing for $$\delta$$, we can then update our solution using the equation,

$\begin{align*} 
\phi_{i+1} = \phi_{i} - \delta_i
\end{align*}$

which was derived in section $$4.1.1.$$ Step six is incorporated in the Newton function below.
"""

# ╔═╡ 8ebea522-f667-4bbf-a292-7db7edacac16
md"""
###### Newton's function
We define a Newton function (MyMultiNewton) which called on the $F,b$ and the initial vector $\phi_0$ at a tolerance level of $1.0e-12$ for an iteration limit of $100$. 
"""

# ╔═╡ 8bd71e8c-dde1-4b52-ad87-e31b84b25a11
function MyMultiNewton(F,b,ϕ₀; tol=1.0e-12, maxit=100)
    result=DiffResults.JacobianResult(ϕ₀)                  #result buffer for Jacobian
	history=Float64[]                                      #history vector 
    ϕ=copy(ϕ₀)
    it=1
    while it<maxit
        ForwardDiff.jacobian!(result,(ϕ)->F(ϕ) - b,ϕ)
        res=DiffResults.value(result)                       #resuidual reuslt
        jac=DiffResults.jacobian(result)                    #Jacobian result   
        δ=jac\res                                           #solve for δ
        ϕ= ϕ - δ                                           #update
        nm=norm(δ)                                         #norm
		push!(history,nm)                                  #push norm to history
        if nm<tol
            return ϕ,history
        end

        it=it+1
    end
    throw("convergence failed")
end

# ╔═╡ 05ad79f3-3f01-4285-888f-7b361e463a4a
md"""

###### Termination Criteria

The termintion criteria is based on the norm and the tolerance level. 

Once the first approximate solution vector $\phi_1$ is computed, the iteration process continous until we obtain convergence to the solution $\phi^{*}$. 

For convergence of a set of vectors, the norm must be zero. That is, 

$$||\phi_{i+1} - \phi_i|| = \sqrt{(\phi_{i+1}^{1} - \phi_{i}^{1})^2 + ... + (\phi_{i+1}^{n} - \phi_{i}^{n})^2} = 0.$$

Hence, Newton's iteration function will terminate when the norm is zero and less than the level of tolerance.
"""

# ╔═╡ 057e088e-7ece-4d4a-9a49-7e1fc3082d1f
md"""
##### Solve System

Now, we shall use the Newton function (MyMultiNewton) to evaluate our system given the nonlinear operator $F$, the right hand side vector $b$ and the inital value $\phi_0$.
"""

# ╔═╡ d312760c-e049-469e-8661-31477769cd59
MyMultiNewton_result,MyMultiNewton_history=MyMultiNewton(F,b,ϕ₀)

# ╔═╡ 119371ce-282a-4bbc-8272-e97feaefc92d
MyMultiNewton_result

# ╔═╡ 3ad46a0b-789f-4cc4-b64e-e89ccc8a7cca
MyMultiNewton_history

# ╔═╡ 567ed50d-b1ba-4b3f-ac9b-03208b411bf3
md"""
It is possible to solve for the residual for our result. This we do by substituting the iteration results: NewtonF into the nonlinear system. 
"""

# ╔═╡ 756d0d07-cf45-497d-88dd-6fab20d7902e
F(MyMultiNewton_result)-b

# ╔═╡ 498e22c2-7c26-4496-b6e3-ecbb1655d768
md""" 
###### Results
Iteration steps: $(length(MyMultiNewton_history))

Using the Newton function, obtained the result for our system $$F(\phi) = b$$ in six iteration steps. The iteration stopped due to the fact that, we obtained convergence as explanied earlier. 
"""

# ╔═╡ 4df9fc81-dc9a-452f-baea-b7b673326c85
md"""
Thus, the approximate solution of the system from our computation is written explicitly as:

$\begin{align*} \phi^{*} = 
\begin{bmatrix}
 0.5 \\ 8.10963e-18 \\ -0.523599
\end{bmatrix}
\end{align*}$
"""

# ╔═╡ 8b9214e9-5746-4ab5-a6d5-f2f7e9aac59e
md"""
###### Convergence Result

We the convergence of Newton's iteration below. We observe from the plot below that in fewer iteration steps, we have a good behaviour of the residual. That is, in $6$ steps, we obtain a quadratic convergence which is desirable.  
"""

# ╔═╡ 45f585af-b7e5-458b-9488-25e68ac217d4
plothistory(MyMultiNewton_history)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"

[compat]
DataFrames = "~1.3.5"
DiffResults = "~1.0.3"
ForwardDiff = "~0.10.30"
LaTeXStrings = "~1.3.0"
PlutoUI = "~0.7.39"
PyPlot = "~2.10.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "0f4e115f6f34bbe43c19751c90a38b2f380637b9"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.3"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "924cdca592bc16f14d2f7006754a621735280b74"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.1.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "6bce52b2060598d8caaed807ec6d6da2a1de949e"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.5"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "336cc738f03e069ef2cac55a104eb823455dca75"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.4"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "1fc929f47d7c151c839c5fc1375929766fb8edcc"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.93.1"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "14c1b795b9d764e1784713941e787e1384268103"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.10.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "bc40f042cfcc56230f781d92db71f0e21496dffd"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.5"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═621b6832-de86-11ec-1ee8-3d4c78a7da67
# ╟─061e7727-fb64-42a9-9555-acc7c2a6e9d5
# ╟─4644cb13-bc49-4a6a-aace-78057353d298
# ╟─c2ccf8e2-235b-4575-adcc-963f76adbf47
# ╟─58c8d6db-5acd-484e-87ee-8b0a3b87e94d
# ╠═c44bc767-537f-455f-9344-3b43c1e7b36f
# ╟─031674ed-4ef7-42a8-b884-ca5ce143bd89
# ╠═d8681c01-a464-46ba-bcd4-a14794492082
# ╠═80074f16-0e6a-4981-b11a-708c38320b3d
# ╠═ffe7267f-3739-417a-8af5-4604289178eb
# ╟─3e32e664-0fc6-46cf-9164-b0f11e351d53
# ╠═ec51b7e2-302a-4fe1-bc01-7bf6996a50a6
# ╟─9679734f-1d76-44f3-8e0a-e365869192b3
# ╟─ef5cfe9f-8462-4a26-b95c-2b3bbe8e07ee
# ╠═cba9bf8d-ea27-484c-9e58-1f424ff04209
# ╟─5788401c-a830-4096-b239-b44a571a459f
# ╟─e2eb4ecd-2233-4be9-884a-fd0f91516391
# ╠═5d9a24d1-246f-43c0-87ec-30090f2b4c4b
# ╠═6057daab-03ba-4a97-9799-3af305854386
# ╠═4e2f13aa-6cc9-44e1-8052-517042489c7b
# ╠═cd3ab5fb-6186-40e6-8731-bb814b4286fe
# ╠═fb49df9b-765d-4a77-8da2-b97c0aed80b5
# ╠═89925314-040d-4dfe-9489-c923a929ab98
# ╟─7d108cb8-0f67-42f3-bb2e-eb580302d8cb
# ╟─b53d1663-eb65-4ef8-93fc-d9f619548ce0
# ╟─8ebea522-f667-4bbf-a292-7db7edacac16
# ╠═8bd71e8c-dde1-4b52-ad87-e31b84b25a11
# ╟─05ad79f3-3f01-4285-888f-7b361e463a4a
# ╟─057e088e-7ece-4d4a-9a49-7e1fc3082d1f
# ╠═5ec2097a-89ac-45f4-85e9-b02cf16283e0
# ╠═d312760c-e049-469e-8661-31477769cd59
# ╠═119371ce-282a-4bbc-8272-e97feaefc92d
# ╠═3ad46a0b-789f-4cc4-b64e-e89ccc8a7cca
# ╟─567ed50d-b1ba-4b3f-ac9b-03208b411bf3
# ╠═756d0d07-cf45-497d-88dd-6fab20d7902e
# ╟─498e22c2-7c26-4496-b6e3-ecbb1655d768
# ╟─4df9fc81-dc9a-452f-baea-b7b673326c85
# ╟─8b9214e9-5746-4ab5-a6d5-f2f7e9aac59e
# ╠═45f585af-b7e5-458b-9488-25e68ac217d4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
