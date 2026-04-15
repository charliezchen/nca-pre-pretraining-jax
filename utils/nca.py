import flax.linen as nn
import jax
from flax.core import freeze, unfreeze

import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
from jax.random import split

import io
import gzip
from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from .tokenizers import Tokenizer

#############################################################
#  NCA Simulation
#############################################################

def rollout_simulation(rng, params, s0=None,
                       substrate=None, fm=None, rollout_steps=256, time_sampling='final', img_size=224, start_step=0,return_state=False, k_steps=1):
    """
    Rollout a simulation described by the specified substrate and parameters.

    Parameters
    ----------
    rng : jax rng seed for the rollout
    params : parameters to configure the simulation within the substrate
    s0 : (optional) initial state of the simulation. If None, then substrate.init_state(rng, params) is used.
    substrate : the substrate object
    fm : the foundation model object. If None, then no image embedding is calculated.
    rollout_steps : number of timesteps to run simulation for
    time_sampling : one of either
        - 'final': return only the final state data (default)
        - 'video': return the entire rollout
        - (K, chunk_ends): return the rollout at K sampled intervals, if chunk_ends is True then end of intervals is sampled
    img_size : image size to render at. Leave at 224 to avoid resizing again for CLIP.
    return_state : return the state data, leave as False, unless you really need it.

    Returns
    ----------
    A dictionary containing
    'state' : the state data of the simulation, None if return_state is False
        shape: (...)
    'rgb' : the image data of the simulation,
        shape (H, W, C)
    'z' : the image embedding of the simulation using the foundation model,
        shape (D)

    If time_sampling is 'video' then the returned shapes become (rollout_steps, ...).
    If time_sampling is an int then the returned shapes become (time_sampling, ...).

    ----------
    This function should be used like this:
    ```
    fm = create_foundation_model('clip')
    substrate = create_substrate('lenia')
    rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=256, time_sampling=8, img_size=224, return_state=False)
    rollout_fn = jax.jit(rollout_fn) # jit for speed
    # now you can use rollout_fn as you need...
    rng = jax.random.PRNGKey(0)
    params = substrate.default_params(rng)
    rollout_data = rollout_fn(rng, params)
    ```

    Note: 
    - when time_sampling is 'final', the function returns data for the T timestep.
    - when time_sampling is 'video', the function returns data for the [0, ..., T-1] timesteps.
    - when time_sampling is (K, False), the function returns data for the [0, T//K, T//K * 2, ..., T - T//K] timesteps.
    - when time_sampling is (K, True), the function returns data for the [T//K, T//K * 2, ..., T] timesteps.
    """

    if s0 is None:
        s0 = substrate.init_state(rng, params)
    
    if time_sampling == 'final': # return only the final state
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            return next_state, None
        state_final, _ = jax.lax.scan(step_fn, s0, split(rng, rollout_steps))
        
        if return_state:
            return jnp.stack([s0, state_final])
        else:
            return state_final
    elif time_sampling == 'video': # return the entire rollout
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            return next_state, state
        _, state_vid = jax.lax.scan(step_fn, s0, split(rng, start_step+rollout_steps))
        idx = jnp.arange(start_step, start_step+rollout_steps, k_steps)
        return state_vid[idx]
    else:
        raise ValueError(f"time_sampling {time_sampling} not recognized")


#############################################################
#  NCA Network adapted from https://github.com/SakanaAI/asal/
#############################################################

class NCANetwork(nn.Module):
    d_state: int = 16
    @nn.compact
    def __call__(self, x):
        x = jnp.pad(x, pad_width=1, mode='wrap')
        x = nn.Conv(features=4, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.Conv(features=16, kernel_size=(1, 1))(x)

        x = nn.relu(x)
        x = nn.Conv(features=self.d_state, kernel_size=(1, 1))(x)
        return x

class RandomMLP(nn.Module):
    n_layers: int
    d_hidden: int
    d_out: int
    activation: Callable
    @nn.compact
    def __call__(self, x, train=False):
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.d_hidden, kernel_init=nn.initializers.normal(1), bias_init=nn.initializers.normal(1))(x)
            x = nn.BatchNorm(use_running_average=not train, momentum=0., use_bias=False, use_scale=False)(x)
            x = self.activation(x)
        x = nn.Dense(features=self.d_out)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0., use_bias=False, use_scale=False)(x)
        return x
        
def create_random_net(net, rng, x):
    rng, _rng = split(rng)
    params = net.init(_rng, jnp.zeros_like(x[0]))
    y, updates = net.apply(params, x, train=True, mutable=['batch_stats'])
    params = unfreeze(params)
    params['batch_stats'] = updates['batch_stats']
    params = freeze(params)
    y = net.apply(params, x)
    assert jnp.allclose(y.mean(axis=0), jnp.zeros_like(y.mean(axis=0)), atol=1e-2)
    assert jnp.allclose(y.std(axis=0), jnp.ones_like(y.std(axis=0)), atol=1e-2)


#############################################################
#  NCA Parameterization
#############################################################

"""
The Discrete NCA substrate.
This substrate models a grid of integer values and updates them using a stochastic neural cellular automata.
"""
class NCA():
    def __init__(self, grid_size=64, d_state=8, n_groups=1, identity_bias=0., temperature=1., color_map='fixed'):
        self.grid_size = grid_size
        self.d_state, self.n_groups = d_state, n_groups
        self.nca = NCANetwork(d_state=d_state*n_groups)

        self.identity_bias, self.temperature = identity_bias, temperature

        self.color_map = color_map
        self.color_palette = 'ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00'
        self.color_palette = jnp.array([mcolors.to_rgb(f"#{a}") for a in self.color_palette.split('-')]) # 8 3

    def default_params(self, rng):
        params = dict()
        if self.color_map != 'fixed':
            rng, _rng = split(rng)
            params['color_map'] = jax.random.normal(_rng, (self.n_groups, self.d_state, 3)) # unconstrainted

        rng, _rng = split(rng)
        params['net_params'] = self.nca.init(_rng, jnp.zeros((self.grid_size, self.grid_size, self.d_state*self.n_groups))) # unconstrainted

        rng, _rng = split(rng)
        params['init'] = jax.random.normal(_rng, (self.n_groups, self.d_state)) # unconstrainted
        return params
    
    def init_state(self, rng, params):
        init = repeat(jax.random.normal(rng, (self.n_groups, self.d_state)), "G D -> H W G D", H=self.grid_size, W=self.grid_size)
        # init = repeat(params['init'], "G D -> H W G D", H=self.grid_size, W=self.grid_size)
        # changing the way we initialize starting states
        state = jax.random.categorical(rng, init, axis=-1)
        return state
    
    def step_state(self, rng, state, params):
        state_oh = jax.nn.one_hot(state, self.d_state)
        state_oh_f = rearrange(state_oh, "H W G D -> H W (G D)")
        logits = self.nca.apply(params['net_params'], state_oh_f)
        logits = rearrange(logits, "H W (G D) -> H W G D", G=self.n_groups)
        
        # identity_bias = jax.nn.sigmoid(params['identity_bias'])*10
        next_state = jax.random.categorical(rng, (logits + state_oh*self.identity_bias)/self.temperature, axis=-1)
        return next_state
    
    def render_state(self, state, params, img_size=None):
        if self.color_map=='fixed':
            img = self.color_palette[state[:, :, 0]]
        else:
            def get_color(color_map, state):
                return color_map[state]
            # color_map: G D 3 # state: H W G
            get_color = jax.vmap(get_color, in_axes=(0, 2))
            img = get_color(jax.nn.sigmoid(params['color_map']), state)
            img = img.mean(axis=0) # average over groups

        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img

#############################################################
#  NCA Dataset Generation
#############################################################

def generate_nca_dataset(
    seed: jax.random.PRNGKey,
    num_sims:int,
    grid: int = 12,
    d_state: int = 10,
    n_groups: int = 1,
    identity_bias: float = 0.,
    temperature: float = 0.,
    num_examples: int = 10,
    num_rules: int = 10,
    dT: int = 2,
    start_step: int = 0,
    rule_seeds = None
):
    generator = NCA(
        grid_size=grid,
        d_state=d_state,
        n_groups=n_groups,
        identity_bias=identity_bias,
        temperature=temperature
    )

    if(rule_seeds is None):
        rule_seeds = jax.random.split(seed, num_rules)
    else:
        n_rules = rule_seeds.shape[0]
        if num_sims <= n_rules:
            # round-robin a contiguous slice so the same sim index repeatedly
            # pairs with the same rule across refills is avoided via the
            # caller reshuffling indices.
            pick = jax.random.randint(seed, (num_sims,), 0, n_rules)
            rule_seeds = rule_seeds[pick]
        else:
            reps = (num_sims + n_rules - 1) // n_rules
            rule_seeds = jnp.tile(rule_seeds, (reps, 1))[:num_sims]

    # generate simulations
    def rollout_fn(rng, task_seed):
        params = generator.default_params(task_seed)
        return rollout_simulation(rng, params, substrate=generator, rollout_steps=dT*num_examples, k_steps=dT, time_sampling='video', start_step=start_step)
        
    sim_data = jax.vmap(rollout_fn, in_axes=(0,0))(split(seed, num_sims), rule_seeds)

    return sim_data

#############################################################
#  Rule Generation
#############################################################

def generate_rules_batch(
    seed: jax.random.PRNGKey,
    num_rules: int,
    tokenizer: Tokenizer = None,
    threshold: float = 40,
    upper_bound: float = None,
    dT: int = 1,
    n_steps:int = 10,
    mode: str = 'shannon', # 'diff' or 'shannon'
    start_step: int = 0,
    grid: int = 12,
    d_state: int = 10,
    identity_bias: float = 0.,
    temperature: float = 0.
):
    current_seed = seed
    current_rules = 0
    rules = []

    if(upper_bound is None):
        upper_bound = float('inf')

    while(current_rules < num_rules):
        seeds = jax.random.split(current_seed, num_rules) # sample more rules to ensure we get enough
        score = compute_rule_gzip_batch(seeds, tokenizer, grid, d_state, identity_bias, temperature, n_steps, dT, start_step, mode)
        idx = jnp.logical_and(score > threshold, score < upper_bound)

        selected_rules = seeds[idx]
        score = score[idx]

        if (len(selected_rules) > 0):
            rules.append(selected_rules[:num_rules - current_rules])
            current_rules += len(selected_rules)
        current_seed = jax.random.split(current_seed, 1)[0]

    return jnp.concat(rules, axis=0)

def compute_rule_gzip_batch(
    seeds,
    tokenizer: Tokenizer,
    grid: int = 12,
    d_state: int = 10,
    identity_bias: float = 0.,
    temperature: float = 0.,
    n_steps: int = 10,
    dT: int = 1,
    start_step: int = 0,
    mode: str = 'gzip'
):
    sims = generate_nca_dataset(
            seeds[0],
            num_sims=seeds.shape[0],
            grid=grid,
            d_state=d_state,
            n_groups=1,
            identity_bias=identity_bias,
            temperature=temperature,
            num_examples=n_steps, # number of examples to rollout
            dT=dT,
            rule_seeds=seeds,
            num_rules=seeds.shape[0],
            start_step = start_step
        )

    B, T, H, W, C = sims.shape

    if(mode == 'diff'):
        diff = (sims[:,2:,...] != sims[:,1:-1,...]).astype(int)
        average_diff = diff.mean(axis=(1)) # average across timesteps
        score = average_diff.mean(axis=tuple(range(1, sims.ndim)))

    elif(mode == 'gzip'):
        import numpy as np
        seq, _ = tokenizer.encode_task(sims)
        grid_len = (H*W)//(tokenizer.patch**2)+2
        seq = seq.reshape(B, -1, grid_len)
        seq = seq[:, :, 1:-1]
        seq_np = np.asarray(seq).reshape(B, -1).astype(np.int32)
        score = []
        for b in range(B):
            byte_data = seq_np[b].tobytes()
            score.append(gzip_complexity(byte_data))
        score = jnp.array(score)
    return score

def gzip_complexity(byte_data: bytes):
    """
    Compute the GZIP compression complexity of a sequence of tokens.
    """
    buf = io.BytesIO()

    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(byte_data)
    compressed_size = len(buf.getvalue())
    original_size = len(byte_data)
    return compressed_size / original_size