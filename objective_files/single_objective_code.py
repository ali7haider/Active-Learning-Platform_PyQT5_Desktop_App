# -*- coding: utf-8 -*-
"""Single Objective Code.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-ylpc0_8XkoXypjgB0Ntn1ZLoEafhyIp
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import torch
import botorch
import gpytorch

tkwargs = {"dtype": torch.double,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.utils.transforms import unnormalize, normalize

from botorch.acquisition.logei import qLogExpectedImprovement

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# plotting dependencies
import matplotlib
from matplotlib import pyplot as plt
# %matplotlib inline

df = pd.read_csv("Synthetic Data 2_Objectives.csv")

df.shape

"""#### Optimization Boundary"""

lower_bounds = torch.tensor([1.0, 1.0, 1.0], **tkwargs) # to be specified by user
upper_bounds = torch.tensor([5.0, 5.0, 5.0], **tkwargs)

problem_bounds = torch.vstack([lower_bounds, upper_bounds])

"""#### Batch size"""

n_var = 3
n_obj = 1

BATCH_SIZE = 3  # to be specified by user

standard_bounds = torch.zeros(2, n_var, **tkwargs)
standard_bounds[1] = 1

random_state= 42
torch.manual_seed(random_state)


# set the tensor according to latest cumulative dataset
train_x = torch.tensor(df.iloc[:,:n_var].to_numpy(), **tkwargs)
train_obj = torch.tensor(df.iloc[:,n_var:n_var+n_obj].to_numpy(), **tkwargs)

"""### Train Gaussian Process"""
# Debug: print training data shapes
print(f"train_x shape: {train_x.shape}")
print(f"train_obj shape: {train_obj.shape}")

train_x_gp = normalize(train_x, problem_bounds)
train_y = train_obj

model = SingleTaskGP(train_x_gp, train_y)

mll = ExactMarginalLogLikelihood(model.likelihood, model)

fit_gpytorch_model(mll);

"""### Acquisition Function"""

sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]), resample=False, seed=0)

acqf = qLogExpectedImprovement(model, best_f = torch.max(model.train_targets), sampler=sampler)

next_x, _ = optimize_acqf(
    acq_function=acqf,
    bounds = standard_bounds,
    q=BATCH_SIZE,
    num_restarts=50,
    raw_samples=512,
    options={"batch_limit": 5, "maxiter": 200},
)

# unnormalize
new_x = unnormalize(next_x.detach(), bounds=problem_bounds)
new_x = new_x.cpu().numpy()

# Generate 50 optimized candidates
df_candidates= pd.DataFrame(new_x, columns=df.columns[1:])
df_candidates = df_candidates.round(2)
print(df_candidates)
