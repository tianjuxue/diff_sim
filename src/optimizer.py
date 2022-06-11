import jax
import jax.numpy as np
from jax import linear_util as lu
from jax.flatten_util import ravel_pytree
import numpy as onp
from functools import partial
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys