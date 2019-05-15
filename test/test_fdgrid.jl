@testset "test grid                              " begin
    # can't do silly things
    @test_throws ArgumentError FDGrid( 0, 3,  -1,  1, 1.0)
    @test_throws ArgumentError FDGrid(10, 3,  -1,  1, 2.0)
    @test_throws ArgumentError FDGrid(10, 3,  -1,  1, 0.0)
    @test_throws ArgumentError FDGrid(10, 3,   1, -1, 0.5)
    @test_throws ArgumentError FDGrid(10, 2,  -1,  1, 0.5)
    @test_throws ArgumentError FDGrid(10, 8,  -1,  1, 0.5)
    @test_throws ArgumentError FDGrid(10, 99, -1,  1, 0.5)

    # uniform distribution
    g = FDGrid(3, 3, -2, 3, 1.0)
    
    # indexing 
    @test g[1] ≈ -2
    @test g[2] ≈  (3-2)/2
    @test g[3] ≈  3

    # for α tending to zero the points converge to the 
    # extrema of the chebychev polynomials to machine  accuracy
    for M in (10, 100, 1000)
        g = FDGrid(M, 5, -1, 1, 1e-100)
        expected = reverse(cos.((0:(M-1))./(M-1).*π))
        @test maximum(abs.(gridpoints(g) - expected)) < 1e-15
    end
end