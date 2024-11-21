# Import packages
using LinearAlgebra

# Variable
x = 2
y = 3

# Artihmetic
z = x + y

# Functions
f(x) = x^2
println("f(3) = $(f(3))")

function g(x, y)
    return x + y
end
println("g(2, 3) = $(g(2, 3))")

# Control Flow
if x > y
    println("x is greater than y")
elseif x < y
    println("x is less than y")
else
    println("x is equal to y")
end

# Loops
for i in 2:2:10  # From 2 to 10 in steps of 2
    println(i)
end

# Matrices
A = [1 2; 3 4]
display(A)
println(A[2, 1])  # 1-indexed

B = A * I  # Matrix multiplication
display(B)
# I is the (dimensionless) identity matrix defined in LinearAlgebra as a struct (UniformScaling).
# Using _multiple dispatch_, the * operator is defined with a special implementation for the UniformScaling struct.
println(length(methods(*)))
display(@which A * I)

C = A .* 2  # Element-wise multiplication (broadcasting)
@info "Broadcasting * 2" A twoA=C

square(x) = x^2
D = square.(A)  # Broadcasting a function
@info "Broadcasting square" A Asquared=D