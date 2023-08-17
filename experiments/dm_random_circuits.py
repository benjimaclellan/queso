from functools import partial
import time

import pandas as pd
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import tensorcircuit as tc
from tensorcircuit.quantum import sample_bin2int, sample_int2bin
import jax
import jax.numpy as jnp

backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("greedy")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”

#%%
n = 4
c = tc.Circuit(n)
