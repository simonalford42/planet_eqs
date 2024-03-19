from modules import BioMLP
from modules import mlp

b = BioMLP(10, 10, 10, 3)
m = mlp(10, 10, 10, 1)

b.linears[0].linear.bias = m[0].bias
b.linears[1].linear.bias = m[2].bias
b.linears[2].linear.bias = m[4].bias

b.linears[0].linear.weight = m[0].weight
b.linears[1].linear.weight = m[2].weight
b.linears[2].linear.weight = m[4].weight

print(b)
print(m)