#!/usr/bin/env python
# coding: utf-8

# # CDF and PDF plots of Standard Normal Distribution

# In[1]:


import os

try:
    import jax
except:
    get_ipython().run_line_magic('pip', 'install jax jaxlib')
    import jax

import jax.numpy as jnp
from jax.scipy.stats import norm

try:
    import matplotlib.pyplot as plt
except:
    get_ipython().run_line_magic('pip', 'install matplotlib')
    import matplotlib.pyplot as plt

try:
    import seaborn as sns
except:
    get_ipython().run_line_magic('pip', 'install seaborn')
    import seaborn as sns


# In[2]:


dev_mode = "DEV_MODE" in os.environ

if dev_mode:
    import sys

    sys.path.append("scripts")
    from plot_utils import latexify, savefig

    latexify(width_scale_factor=2, fig_height=1.5)


# In[3]:


# Plot pdf and cdf of standard normal

x = jnp.linspace(-3, 3, 500)
random_var = norm


def make_graph(data, save_name):
    fig, ax = plt.subplots()
    ax.plot(data["x"], data["y"])
    # plt.title("Gaussian pdf")
    plt.xlabel("x")
    plt.ylabel(data["ylabel"])
    sns.despine()
    if dev_mode:
        savefig(save_name)


make_graph({"x": x, "y": random_var.pdf(x), "ylabel": "$p(x)$"}, "gaussian1d_latexified.pdf")
make_graph(
    {"x": x, "y": random_var.cdf(x), "ylabel": "$Pr(X \leq x)$"},
    "gaussianCdf_latexified.pdf",
)

