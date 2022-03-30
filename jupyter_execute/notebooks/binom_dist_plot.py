#!/usr/bin/env python
# coding: utf-8

# # Plots the pmfs of binomial distributions with varying probability of success parameter

# In[1]:


import os

try:
    import jax
except:
    get_ipython().run_line_magic('pip', 'install jax jaxlib')
    import jax
import jax.numpy as jnp
from jax.scipy.stats import nbinom

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
    from scipy.stats import binom
except:
    get_ipython().run_line_magic('pip', 'install scipy')
    from scipy.stats import binom


# In[2]:


dev_mode = "DEV_MODE" in os.environ

if dev_mode:
    import sys

    sys.path.append("scripts")
    from plot_utils import latexify, savefig

    latexify(width_scale_factor=2, fig_height=1.5)


# In[3]:


N = 10
thetas = [0.25, 0.5, 0.75, 0.9]
x = jnp.arange(0, N + 1)


def make_graph(data):
    plt.figure()
    x = data["x"]
    n = data["n"]
    theta = data["theta"]

    probs = binom.pmf(x, n, theta)
    title = r"$\theta=" + str(theta) + "$"

    plt.bar(x, probs, align="center")
    plt.xlim([min(x) - 0.5, max(x) + 0.5])
    plt.ylim([0, 0.4])
    plt.xticks(x)
    plt.xlabel("$x$")
    plt.ylabel("$p(x)$")
    plt.title(title)
    sns.despine()
    if dev_mode:
        savefig("binomDistTheta" + str(int(theta * 100)) + "_latexified.pdf")


for theta in thetas:
    data = {"x": x, "n": N, "theta": theta}
    make_graph(data)


# ## Demo
# You can see different examples of binomial distributions by changing the theta in the following demo.

# In[4]:


from ipywidgets import interact


@interact(theta=(0.1, 0.9))
def generate_random(theta):
    n = 10
    data = {"x": jnp.arange(0, n + 1), "n": n, "theta": theta}
    make_graph(data)

