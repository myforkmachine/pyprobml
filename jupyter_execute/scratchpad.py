#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
# 2x 2 chips (4 cores) per process:
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,2,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
# Different per process:
os.environ["TPU_VISIBLE_DEVICES"] = "0,1" # Change to "2,3" for the second machine
# Pick a unique port per process
os.environ["TPU_MESH_CONTROLLER_ADDRESS"] = "localhost:8476"
os.environ["TPU_MESH_CONTROLLER_PORT"] = "8476"

print('done')


# In[11]:


import jax
print(jax.devices())#


# In[ ]:


# Check that RNG works
# Context: https://github.com/google/jax/issues/7896
import jax
import jax.numpy as jnp

# sample from a Markov chain
init_dist = jnp.array([0.8, 0.2])
trans_mat = jnp.array([[0.9, 0.1], [0.5, 0.5]])
rng_key = jax.random.PRNGKey(0)
from jax.scipy.special import logit
seq_len = 15

initial_state = jax.random.categorical(rng_key, logits=logit(init_dist), shape=(1,))

def draw_state(prev_state, key):
        logits = logit(trans_mat[:, prev_state])
        state = jax.random.categorical(key, logits=logits.flatten(), shape=(1,))
        return state, state

        rng_key, rng_state, rng_obs = jax.random.split(rng_key, 3)
        keys = jax.random.split(rng_state, seq_len - 1)

        final_state, states = jax.lax.scan(draw_state, initial_state, keys)

        print(states)
            
rng_key, rng_state, rng_obs = jax.random.split(rng_key, 3)
keys = jax.random.split(rng_state, seq_len - 1)

final_state, states = jax.lax.scan(draw_state, initial_state, keys)

print(states)


# In[13]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[12]:


get_ipython().run_line_magic('cd', '~/base/project1/github')


# In[14]:


get_ipython().run_line_magic('run', 'pyprobml/scripts/ab_test_demo.py')


# In[15]:


from jsl.demos import logreg_biclusters_demo as demo
figures, data = demo.main()
print(data)


# In[7]:


get_ipython().run_line_magic('run', 'JSL/jsl/demos/eekf_logistic_regression_demo.py')


# In[16]:


get_ipython().run_line_magic('run', 'JSL/jsl/demos/logreg_biclusters_demo.py')


# In[5]:


get_ipython().run_line_magic('run', 'pyprobml/scripts/vb_gauss_cholesky_biclusters_demo.py')


# In[27]:


get_ipython().run_line_magic('run', 'pyprobml/scripts/kf_tracking_demo.py')

