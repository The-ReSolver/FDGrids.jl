export gridpoints

function gridpoints(M::Int, l::Real=-1.0, h::Real=1.0, α::Real=0.5)
    0 < α ≤ 1      || throw(ArgumentError("α must ∈ (0, 1]"))
    M > 1          || throw(ArgumentError("M must be at least two"))
    l < h          || throw(ArgumentError("l must be lower than h"))

    # define grid points
    j = 0:(M-1)
    xs = asin.(.-α.*cos.(π.*j./(M.-1)))./asin.(α).*(h.-l)./2 .+ (h.+l)./2

    return xs
end