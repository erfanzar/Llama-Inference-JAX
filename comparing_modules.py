import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
from src.llama_interface_jax.covertors.convert_llama import pt2jax
from src.llama_interface_jax.model import LiJAXLinear, LiJAXEmbed
from torch import nn
import torch


def comparing_linear():
    x = torch.randn((1, 5))
    linear = nn.Linear(5, 10, bias=False)
    uq_lijax_linear = LiJAXLinear.from_torch(linear, quantize=False)
    q_lijax_linear = LiJAXLinear.from_torch(linear, quantize=True)
    print(f"TORCH    LINEAR : {linear}")
    print(f"UQ LiJAX LINEAR : {uq_lijax_linear}")
    print(f"Q  LiJAX LINEAR : {q_lijax_linear}")
    torch_out = linear(x)
    uq_lijax_out = uq_lijax_linear(pt2jax(x))
    q_lijax_out = q_lijax_linear(pt2jax(x))
    print(f"TORCH    M : {pt2jax(torch_out.sum().mean())}")
    print(f"UQ LiJAX M : {uq_lijax_out.sum().mean()}")
    print(f"Q  LiJAX M : {q_lijax_out.sum().mean()}")


def comparing_embedding():
    dim = 32000
    x = torch.randint(0, dim, (1, 5), )
    embed = nn.Embedding(dim, 10)
    uq_lijax_linear = LiJAXEmbed.from_torch(embed, quantize=False)
    q_lijax_linear = LiJAXEmbed.from_torch(embed, quantize=True)
    print(f"TORCH    Embed : {embed}")
    print(f"UQ LiJAX Embed : {uq_lijax_linear}")
    print(f"Q  LiJAX Embed : {q_lijax_linear}")

    torch_out = embed(x)
    uq_lijax_out = uq_lijax_linear(pt2jax(x))
    q_lijax_out = q_lijax_linear(pt2jax(x))

    print(f"TORCH    M : {pt2jax(torch_out.sum().mean())}")
    print(f"UQ LiJAX M : {uq_lijax_out.sum().mean()}")
    print(f"Q  LiJAX M : {q_lijax_out.sum().mean()}")


if __name__ == "__main__":
    comparing_linear()
    print("*" * 20)
    comparing_embedding()
