import numpy as np
import pandas as pd
import torch
import warnings
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

# Set print options
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
torch.set_printoptions(precision=3)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set tensor properties
tkwargs = {"dtype": torch.double, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}

try:
    print("Loading dataset...")
    df = pd.read_csv("Synthetic Data 2_Objectives.csv")
    print("Dataset loaded successfully!")
    print(df.head())

    # Define problem boundaries
    lower_bounds = torch.tensor([1, 1, 1, 1], **tkwargs)
    upper_bounds = torch.tensor([10, 5, 5, 5], **tkwargs)
    problem_bounds = torch.vstack([lower_bounds, upper_bounds])

    # Set variables and objectives
    n_var = 4
    n_obj = 2
    random_state = 42
    torch.manual_seed(random_state)

    # Convert dataset to tensors
    print("Converting dataset to tensors...")
    train_x = torch.tensor(df.iloc[:, :n_var].to_numpy(), **tkwargs)
    train_obj = torch.tensor(df.iloc[:, n_var:n_var + n_obj].to_numpy(), **tkwargs)
    print("Tensor conversion completed!")

    # Normalize input data
    print("Normalizing input data...")
    standard_bounds = torch.zeros(2, train_x.shape[1], **tkwargs)
    standard_bounds[1] = 1
    train_x_norm = normalize(train_x, problem_bounds)

    # Train Gaussian Process models
    print("Training GP models...")
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        models.append(SingleTaskGP(train_x_norm, train_y, outcome_transform=Standardize(m=1)))

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    print("Fitting GP model...")
    try:
        from botorch.fit import fit_gpytorch_model
        fit_gpytorch_model(mll)
    except ImportError:
        print("Using alternative fit method due to import error.")
        optimizer = torch.optim.Adam([{"params": mll.parameters()}], lr=0.1)
        for _ in range(50):
            optimizer.zero_grad()
            output = mll.model(mll.model.train_inputs[0])
            loss = -mll(output, mll.model.train_targets)
            loss.backward()
            optimizer.step()

    print("GP model training complete!")

    # Acquisition function
    print("Defining acquisition function...")
    ref_point = torch.tensor([0, 0], **tkwargs)
    BATCH_SIZE = 5
    NUM_RESTARTS = 10
    RAW_SAMPLES = 128

    from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_x_norm,
        sampler=SobolQMCNormalSampler(torch.Size([512])),

        objective=IdentityMCMultiOutputObjective(outcomes=np.arange(n_obj).tolist()),
        prune_baseline=True,
        cache_pending=True
    )

    print("Running optimization...")
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        q=BATCH_SIZE,
        bounds=standard_bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200}
    )

    # Unnormalize results
    print("Unnormalizing results...")
    new_x = unnormalize(candidates.detach(), bounds=problem_bounds).cpu().numpy()
    df_candidates = pd.DataFrame(new_x, columns=df.columns[:4]).round(2)

    print("Optimization complete! Here are the generated candidates:")
    print(df_candidates)

except Exception as e:
    print(f"An error occurred: {e}")
