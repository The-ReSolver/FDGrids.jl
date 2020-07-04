using InteractiveUtils
using BenchmarkTools
using LinearAlgebra
using FDGrids

M = 300
xs = gridpoints(M, -1, 1, 0.5)
width = 3

# diffmatrix
D = DiffMatrix(xs, width, 2)
D[1,   :] .= [1, zeros(M-1)...]
D[end, :] .= [zeros(M-1)..., 1];

# random right hand side
b = randn(M)

# old routines with pivoting
luDdiff = lu(D)
DC = banded_lu!(copy(D))

@btime ldiv!($luDdiff, bc) setup=(bc = copy(b))
@btime LinearAlgebra.ldiv!(DC_, bc) setup=(DC_=copy(DC); bc = copy(b))
@btime LinearAlgebra.ldiv!(DC_, bc, $width-1, $width-1) setup=(DC_=copy(DC); bc = copy(b))

