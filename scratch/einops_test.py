#%%
from einops import rearrange, reduce, repeat
import jax.numpy as jnp

#%%
input_tensor = jnp.arange(500).reshape(2, 50, 5).astype('float')
print(input_tensor)

#%%
# rearrange elements according to the pattern
output_tensor = rearrange(input_tensor, 'b m q -> m b q')
print(output_tensor)

#%%
print(reduce(input_tensor, 'b m q -> b m', 'max'))

#%%
a = jnp.arange(20).reshape(2, 10).astype('float32')
print(a)
#%%
print(repeat(a, 'b s -> b s c', c=2))



#%%

# # combine rearrangement and reduction
# output_tensor = reduce(input_tensor, 'b c (h h2) (w w2) -> b h w c', 'mean', h2=2, w2=2)
# # copy along a new axis
# output_tensor = repeat(input_tensor, 'h w -> h w c', c=3)
