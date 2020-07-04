module FDGrids

import LinearAlgebra

include("utils.jl")
include("diffmatrix.jl")
include("linalg.jl")
include("matmul.jl")
include("grids.jl")


export DiffMatrix, gridpoints, full, banded_lu!, banded_tril_solve!, banded_triu_solve!, banded_solve!

end