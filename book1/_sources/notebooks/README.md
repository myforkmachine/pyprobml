# Notebooks

## Instructions for developers

### Generating figures and saving them locally
* To generate pdf figures, go to the root of the repo and use the following command:
```py
DEV_MODE=1 ipython <notebook>.ipynb
```
For example:
```py
DEV_MODE=1 ipython notebooks/discrete_prob_dist_plot.ipynb
```

* Figures will be saved in `figures` folder in the root of the repo by default.
* Running the notebooks without the `DEV_MODE` flag will not save any figures and is equivalant to executing the notebooks normally (from GUI).