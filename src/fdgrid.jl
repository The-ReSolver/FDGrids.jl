export FDGrid,
       gridpoints

struct FDGrid{T, WIDTH}
    xs::Vector{T}            # the location of the grid points
    D1::DiffMatrix{T, WIDTH} # diff matrix for first derivative
    D2::DiffMatrix{T, WIDTH} # diff matrix for second derivative
    function FDGrid(M::Int,
                width::Integer=9,
                    l::Real=-1.0,
                    h::Real=1.0,
                    α::Real=1.0,
                     ::Type{T}=Float64) where {T}
        0 < α ≤ 1      || throw(ArgumentError("α must ∈ (0, 1]"))
        M > 0          || throw(ArgumentError("M must be positive"))
        l < h          || throw(ArgumentError("l must be lower than h"))
        3 ≤ width ≤ 17 || throw(ArgumentError("stencil width must be between 3 and 17"))
        width % 2 == 1 || throw(ArgumentError("stencil width must be odd"))

        # define grid points
        j = 0:(M-1)
        xs = asin.(.-α.*cos.(π.*j./(M.-1)))./asin.(α).*(h.-l)./2 .+ (h.+l)./2

        # instantiate
        new{T, width}(xs, DiffMatrix(xs, width, 1), DiffMatrix(xs, width, 2))
    end
end

"""
    gridpoints(g::FDGrid)

Returns the grid points of the grid.
"""
gridpoints(g::FDGrid) = g.xs

"""
    Base.getindex(g::FDGrid, i::Integer)

Returns the location of the `i`-the grid points
"""
Base.getindex(g::FDGrid, i::Integer) = g.xs[i]