# test_julia.jl
function sum_loop(n)
    total = 0
    for i = 0:n-1  # range(n) in Python is 0 to n-1
        total += 1/(1+i)
    end
    return total
end

@time result = sum_loop(10000000)
println("Julia result: ", result)