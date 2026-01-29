from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add
# Graph fragment:
#   %arg0_1 : Tensor "f32[1024][1]cpu" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1024][1]cpu" = PlaceHolder[target=arg1_1]
#   %add : Tensor "f32[1024][1]cpu"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   return %add
import functools
import math
import torch
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from torch._inductor.runtime.runtime_utils import pallas_partial_reduce, torch_dtype_to_jax_runtime

from jax.experimental.pallas import mosaic_gpu as plgpu
def pallas_fused_add_6bd44fe3_kernel(#out_ptr0_alias,
        in_ptr0, in_ptr1, out_ptr0):
    # Define iteration variables as JAX arrays
    # x0 = jnp.arange(1024)
    tmp0 = in_ptr0[...]
    tmp1 = in_ptr1[...]
    tmp2 = tmp0 + tmp1
    # out_ptr0[...] = (jnp.full(out_ptr0.shape, tmp2) if jnp.asarray(tmp2).ndim == 0 else (jnp.broadcast_to(jnp.asarray(tmp2), out_ptr0.shape) if jnp.asarray(tmp2).size != out_ptr0.size else jnp.asarray(tmp2).reshape(out_ptr0.shape)))
    out_ptr0[...] = tmp2
# @functools.partial(jax.jit, static_argnums=(0, 1,), donate_argnums=(2,))
def pallas_fused_add_6bd44fe3_jit_wrapper(out_shapes, out_dtypes, out_ptr0_alias, in_ptr0, in_ptr1):
    out_shapes_pallas = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in zip(out_shapes, out_dtypes)
    )
    indexer = lambda n: lambda i: (i,)
    out_specs_pallas = tuple(
        pl.BlockSpec(shape, indexer(len(shape)))
        for shape, dtype in zip(out_shapes, out_dtypes)
    )
    in_specs_pallas = tuple(
        pl.BlockSpec(i.shape, indexer(len(i.shape)))
        for i in [#out_ptr0_alias, 
        in_ptr0, in_ptr1]
    )
    return pl.pallas_call(
        pallas_fused_add_6bd44fe3_kernel,
        out_shape=out_shapes_pallas,
        # out_specs=out_specs_pallas,
        # in_specs=in_specs_pallas,
        interpret=True,
        grid=(1,),
        # input_output_aliases={ 0: 0 },
    )(
        #out_ptr0_alias, 
        in_ptr0, in_ptr1,
    )
def pallas_fused_add_6bd44fe3_main(out_ptr0_alias, in_ptr0, in_ptr1, out_ptr0, stream=None):
    # Enable JAX x64 mode for float64/int64 support
    jax.config.update('jax_enable_x64', True)
    jax.clear_caches()
    # Convert Torch -> JAX for donated outputs
    out_ptr0_alias_jax = jax.device_put(out_ptr0_alias.cpu().numpy(), device=jax.devices('tpu')[0])
    # Convert Torch -> JAX for in-place tensors
    # Convert Torch -> JAX for inputs
    in_ptr0_jax = jax.device_put(in_ptr0.cpu().numpy(), device=jax.devices('tpu')[0])
    in_ptr1_jax = jax.device_put(in_ptr1.cpu().numpy(), device=jax.devices('tpu')[0])
    # Prepare output metadata from PyTorch tensor
    out_shapes = (tuple(out_ptr0.shape),)
    out_dtypes = (torch_dtype_to_jax_runtime(out_ptr0.dtype),)
    res, = pallas_fused_add_6bd44fe3_jit_wrapper(out_shapes, out_dtypes, out_ptr0_alias_jax, in_ptr0_jax, in_ptr1_jax)
    out_jax = jax.device_get(res)
    out_dl = torch.from_numpy(out_jax)
    out_ptr0.copy_(out_dl)

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1024, ), (1, ))
        assert_size_stride(arg1_1, (1024, ), (1, ))
        buf0 = empty_strided_cpu((1024, ), (1, ), torch.float32)
        pallas_fused_add_6bd44fe3_main(buf0, arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    # arg0_1 = 2 + 0 * rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    # arg1_1 = 3 + 0 * rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg0_1 = 2 * torch.ones(1024, device='cpu', dtype=torch.float32)
    arg1_1 = 3 * torch.ones(1024, device='cpu', dtype=torch.float32)
    print(call([arg0_1, arg1_1]))
    # from torch._inductor.wrapper_benchmark import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module)
