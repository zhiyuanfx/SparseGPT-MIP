# Formulation of Pruning as MIQP

We aim to prune a row of a neural network weight matrix using a quadratic formulation.

## Original Loss Formulation

The mean squared error (MSE) loss for approximating outputs with pruned weights can be written as:

$$
\text{Loss} = \frac{1}{N} \| X (z \odot w + e) - y \|_2^2
$$

where:
- $X \in \mathbb{R}^{N \times d}$ is the input matrix,
- $y \in \mathbb{R}^N$ is the output vector,
- $w \in \mathbb{R}^d$ are the original weights,
- $z \in \{0,1\}^d$ are binary mask variables (1 = keep weight, 0 = prune weight),
- $e \in \mathbb{R}^d$ are continuous adjustment variables,
- $N$ is the number of samples.

## Expanding the Loss

Recall that the effective pruned weight vector is:

$$
\tilde{w} = z \odot w + e
$$

where $z \in \{0,1\}^d$ is the binary mask, $w \in \mathbb{R}^d$ are the original weights, and $e \in \mathbb{R}^d$ are continuous error adjustments.  
We concatenate $z$ and $e$ into a single decision vector:

$$
x = \begin{bmatrix} z \\ e \end{bmatrix} \in \mathbb{R}^{2d}.
$$

We then define a mapping matrix $M \in \mathbb{R}^{d \times 2d}$ such that:

$$
M x = z \odot w + e.
$$

Concretely,

$$
M = \big[ \, \mathrm{diag}(w) \ \ \ I_d \, \big],
$$

so that $M x = \mathrm{diag}(w) z + e = z \odot w + e$.

Substituting $\tilde{w} = Mx$ into the mean squared error loss:

$$
\text{Loss} = \frac{1}{N} \| X (Mx) - y \|_2^2
$$

and expanding:

$$
\text{Loss} = \frac{1}{N} \left[ (Mx)^T X^T X (Mx) - 2 (X^T y)^T (Mx) + y^T y \right].
$$

## Transformation into Quadratic Program

We re-express the loss in standard quadratic programming form:

$$
\min_x \ \tfrac{1}{2} x^T Q x + q^T x
$$

where:
- $Q = \frac{2}{N} M^T X^T X M$,
- $q = \frac{-2}{N} M^T X^T y$,
- Constant term $\frac{1}{N} y^T y$ is dropped (does not affect optimization).

## Constraints

### 1. Error Bound Constraints

We require that error adjustments $e_i$ remain bounded relative to the weight magnitudes (e.g., $\rho = 1$):

$$
- z_i |w_i| \rho \;\; \le \;\; e_i \;\; \le \;\; z_i |w_i| \rho, \quad \forall i.
$$

This ensures $e_i = 0$ whenever $z_i = 0$, i.e., no adjustment is allowed if the weight is pruned.  

These constraints can be expressed in matrix form:

$$
A x \le b
$$

where
- $x = \begin{bmatrix} z \\ e \end{bmatrix} \in \mathbb{R}^{2d}$,  
- $A \in \mathbb{R}^{2d \times 2d}$ encodes the inequalities,  
- $b = 0$.  

#### Example (for a single index $i$)

Each variable $i$ contributes **two rows** to $A$:

1. **Upper bound**:  
   $e_i - |w_i| \rho \, z_i \le 0$  

   → in row form (nonzeros only):
   $$
   A[i, \; z_i] = -|w_i|\rho, \quad A[i, \; e_i] = 1.
   $$

2. **Lower bound**:  
   $-e_i - |w_i| \rho \, z_i \le 0$  

   → in row form:
   $$
   A[d+i, \; z_i] = -|w_i|\rho, \quad A[d+i, \; e_i] = -1.
   $$

#### Concretely

Suppose $d=4$, and we look at index $i=2$ (second weight). Then $x = [z_1, z_2, z_3, z_4, e_1, e_2, e_3, e_4]$.  

The two rows for $i=2$ in $A$ would look like:

$$
\begin{bmatrix}
0 & -|w_2|\rho & 0 & 0 & 0 & 1 & 0 & 0 \\[6pt]
0 & -|w_2|\rho & 0 & 0 & 0 & -1 & 0 & 0
\end{bmatrix}.
$$

All other entries are zero.  

So in the full $A$, stacking across all $i=1,\dots,d$, we get a **block structure** where each pair of rows enforces the bounds for its corresponding $(z_i, e_i)$ pair.

### 2. Sparsity Constraints

We enforce that only a subset of weights remain after pruning, using binary variables $z_i \in \{0,1\}$.

#### (a) Unstructured Sparsity

Exactly $k = \lfloor d \cdot s \rfloor$ weights are kept:

$$
\sum_{i=1}^d z_i = k.
$$

In matrix form:

$$
C x = d,
$$

with $C \in \mathbb{R}^{1 \times 2d}$ and $x = [z; e]$.  
Here only the $z$-block matters, so:

$$
C = \begin{bmatrix} 1 & 1 & \dots & 1 \;\;|\;\; 0 & 0 & \dots & 0 \end{bmatrix}, \ d = k.
$$

#### (b) Semi-structured $n:m$ Sparsity

In every group of $m$, exactly $n$ weights are kept:

$$
\sum_{j=1}^m z_{g,j} = n, \quad \forall \ \text{groups } g.
$$

If $d$ is not divisible by $m$, leftover variables are *forced to be kept* (i.e., $z_i = 1$).

#### Example (with $d=6$, $n=2$, $m=3$)

Decision vector:

$$
x = [z_1, z_2, z_3, z_4, z_5, z_6, e_1, e_2, e_3, e_4, e_5, e_6].
$$

- Group 1: $(z_1, z_2, z_3)$  
  Constraint: $z_1 + z_2 + z_3 = 2$  

  Row in $C$:
  $$
  [1 \;\; 1 \;\; 1 \;\; 0 \;\; 0 \;\; 0 \;|\; 0 \dots 0]
  $$

- Group 2: $(z_4, z_5, z_6)$  
  Constraint: $z_4 + z_5 + z_6 = 2$  

  Row in $C$:
  $$
  [0 \;\; 0 \;\; 0 \;\; 1 \;\; 1 \;\; 1 \;|\; 0 \dots 0]
  $$

So the full $C$ matrix (only $z$ block nonzero) looks like:

$$
C =
\begin{bmatrix}
1 & 1 & 1 & 0 & 0 & 0 & | & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 1 & 1 & | & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix},
\quad
d =
\begin{bmatrix}
2 \\
2
\end{bmatrix}.
$$

#### (c) Leftover Variables

If $d$ is not divisible by $m$ (say $d=7$, $m=3$), then the last group has a remainder of 1 element (e.g., $z_7$).  
In this case we enforce:

$$
z_7 = 1,
$$

so the leftover is **always kept**.  

In matrix form, this adds one extra row to $C$:

$$
[0 \;\; 0 \;\; 0 \;\; 0 \;\; 0 \;\; 0 \;\; 1 \;|\; 0 \dots 0],
\quad d = 1.
$$

## QHD MIQP Formulation

We then pass the problem into the `PhiSolve` framework as an MIQP:

- Quadratic objective: $(Q, q)$
- Inequality constraints: $(A, b)$
- Equality constraints: $(C, d)$
- Variable bounds: $0 \le z \le 1$, $- |w_i| \le e_i \le |w_i|$
- First $d$ variables ($z$) are binary, rest ($e$) continuous.

## Summary

The pruning problem is formulated as:

$$
\begin{aligned}
\min_{z \in \{0,1\}^d, e \in \mathbb{R}^d} \quad & \tfrac{1}{2} x^T Q x + q^T x \\
\text{s.t.} \quad 
& - z_i |w_i| \rho \le e_i \le z_i |w_i| \rho, \quad \forall i \\
& \sum_{i=1}^d z_i = k \quad \text{(unstructured)} \\
& \sum_{j=1}^m z_{g,j} = n, \quad \forall g \quad \text{(semi-structured)} \\
& z_i \in \{0,1\}, \ e_i \in [-|w_i|, |w_i|]
\end{aligned}
$$

This MIQP can then be solved with **Gurobi** or the **QHD solver**.