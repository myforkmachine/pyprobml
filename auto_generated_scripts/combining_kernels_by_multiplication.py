#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    import jax
except:
    get_ipython().run_line_magic('pip', 'install jax jaxlib')
    import jax
import jax.numpy as jnp

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

try:
    import tinygp
except ModuleNotFoundError:
    get_ipython().run_line_magic('pip', 'install -qqq tinygp')
    import tinygp
kernels = tinygp.kernels
from tinygp import GaussianProcess


# In[2]:


import os

dev_mode = "DEV_MODE" in os.environ

if dev_mode:
    import sys

    sys.path.append("scripts")
    from plot_utils import latexify, savefig

    latexify(width_scale_factor=4, fig_height=2)


# In[3]:


def make_graph(data, save_name):
    if dev_mode:
        fig, ax = plt.subplots(2, 1)
    else:
        fig, ax = plt.subplots(2, 1, figsize=(6.4, 6))

    # Plot kernel
    kernel = data["kernel1"] * data["kernel2"]
    x2 = jnp.array([1.0]).reshape(-1, 1)
    kernel_values = kernel(x, x2)
    ax[0].plot(x.ravel(), kernel_values.ravel(), color="k")

    # Plot samples
    gp = GaussianProcess(kernel, x)
    samples = gp.sample(key, (2,))
    for sample in samples:
        ax[1].plot(x, sample)
    ax[0].set_title(data["title"])
    ax[1].set_xlabel(data["xlabel"])
    for axes in ax:
        axes.set_xticks([])
    ax[0].set_xlabel("$x$ (with $x'=1$)")
    plt.tight_layout()
    sns.despine()
    if dev_mode and len(save_name) > 0:
        savefig(save_name)
    return fig, ax


x = jnp.arange(-3.0, 5.1, 0.1).reshape(-1, 1)
N = len(x)

key = jax.random.PRNGKey(4)

fig, ax = make_graph(
    {
        "kernel1": kernels.Polynomial(order=1),
        "kernel2": kernels.Polynomial(order=1),
        "title": "Lin x Lin",
        "xlabel": "quadratic functions",
    },
    save_name="kernel_mul_lin_lin_latexified.pdf",
)

fig, ax = make_graph(
    {
        "kernel1": kernels.ExpSquared(scale=4.0),
        "kernel2": kernels.ExpSineSquared(scale=2.0, gamma=0.5),
        "title": "SE x Per",
        "xlabel": "locally periodic",
    },
    save_name="kernel_mul_se_per_latexified.pdf",
)

fig, ax = make_graph(
    {
        "kernel1": kernels.Polynomial(order=1),
        "kernel2": kernels.ExpSquared(scale=1.0),
        "title": "Lin x SE",
        "xlabel": "increasing variation",
    },
    save_name="kernel_mu_lin_se_latexified.pdf",
)

fig, ax = make_graph(
    {
        "kernel1": kernels.Polynomial(order=1),
        "kernel2": kernels.ExpSineSquared(scale=2.0, gamma=1.0),
        "title": "Lin x Per",
        "xlabel": "growing amplitude",
    },
    save_name="kernel_mul_lin_per_latexified.pdf",
)

