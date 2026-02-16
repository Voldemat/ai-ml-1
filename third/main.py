import torch

_ = torch.manual_seed(42)

scalar = torch.tensor(1)
print(f"{scalar=}")
vector = torch.tensor([1, 2, 3])
print(f"{vector=}")
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
print(f"{matrix=}")
transposed = matrix.T
print(f"{transposed=}")

linear = torch.nn.Linear(in_features=3, out_features=3)
output = linear(matrix)
print(f"linear output: {output}")

x = torch.arange(0, 100, 10)
print(f"{x=}")
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
print(f"Mean: {x.type(torch.float32).mean()}")
print(f"Sum: {x.sum()}")

print(f"Index where max value occurs: {x.argmax()}")
print(f"Index where min value occurs: {x.argmin()}")
