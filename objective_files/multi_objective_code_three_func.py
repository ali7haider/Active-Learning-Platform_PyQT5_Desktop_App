import pandas as pd
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.objective import IdentityMCMultiOutputObjective
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

def run_multi_objective_optimization_three(
    df: pd.DataFrame,
    lower_bounds: list,
    upper_bounds: list,
    n_var: int,
    n_obj: int,
    batch_size: int,
    ref_point: list,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Runs the multi-objective optimization process using the provided data and parameters.

    Args:
        df (pd.DataFrame): Input dataset with feature columns and target columns (multiple objectives).
        lower_bounds (list): Lower bounds for the optimization variables.
        upper_bounds (list): Upper bounds for the optimization variables.
        n_var (int): Number of variables (features) to consider.
        n_obj (int): Number of objectives (target columns).
        batch_size (int): Batch size for the optimization.
        ref_point (list): Reference point for the Hypervolume Improvement acquisition function.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing the optimized candidates.
    """
    # Debug: print the starting of the optimization process
    print("Running multi-objective optimization process...")

    # Ensure bounds are numeric
    lower_bounds = [float(b) for b in lower_bounds]
    upper_bounds = [float(b) for b in upper_bounds]

    # Define tensor properties
    tkwargs = {"dtype": torch.double, "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

    # Define bounds as tensors
    lower_bounds_tensor = torch.tensor(lower_bounds, **tkwargs)
    upper_bounds_tensor = torch.tensor(upper_bounds, **tkwargs)
    problem_bounds = torch.vstack([lower_bounds_tensor, upper_bounds_tensor])

    # Normalize standard bounds for acquisition function
    standard_bounds = torch.zeros(2, n_var, **tkwargs)
    standard_bounds[1] = 1

    # Set random seed
    torch.manual_seed(random_state)

    # Prepare training data
    train_x = torch.tensor(df.iloc[:, :n_var].to_numpy(), **tkwargs)
    train_obj = torch.tensor(df.iloc[:, n_var:n_var + n_obj].to_numpy(), **tkwargs)

    # Normalize inputs and define the model
    train_x_gp = normalize(train_x, problem_bounds)
    models = []

    # Train separate GP models for each objective
    for i in range(n_obj):
        train_y = train_obj[..., i:i + 1]
        models.append(SingleTaskGP(train_x_gp, train_y, outcome_transform=Standardize(m=1)))

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    # Train Gaussian Process models
    fit_gpytorch_model(mll)

    # Define acquisition function
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=torch.tensor(ref_point),
        X_baseline=train_x_gp,
        sampler=SobolQMCNormalSampler(torch.Size([512])),
        objective=IdentityMCMultiOutputObjective(outcomes=np.arange(n_obj).tolist()),
        prune_baseline=True,
        cache_pending=True
    )

    # Optimize acquisition function to find the next candidates
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        q=batch_size,
        bounds=standard_bounds,
        num_restarts=10,
        raw_samples=128,
        options={"batch_limit": 5, "maxiter": 200}
    )

    # Unnormalize and return the optimized candidates
    new_x = unnormalize(candidates.detach(), bounds=problem_bounds)
    new_x = new_x.cpu().numpy()

    # Generate optimized candidates DataFrame
    df_candidates = pd.DataFrame(new_x, columns=df.columns[:n_var])
    df_candidates = df_candidates.round(2)

    # Debug: print first few rows of the resulting dataframe
    print("Optimized candidates:")
    print(df_candidates.head())

    return df_candidates
