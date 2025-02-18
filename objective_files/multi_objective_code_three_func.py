import pandas as pd
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import numpy as np
import warnings
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective

warnings.filterwarnings('ignore', category=RuntimeWarning)
tkwargs = {"dtype": torch.double, "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

def run_multi_objective_optimization_three(
    df: pd.DataFrame,
    lower_bounds: list,
    upper_bounds: list,
    n_var: int,
    target_columns: list,
    batch_size: int,
    reference_point: list,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Runs the multi-objective optimization process using the provided data and parameters.
    """
    print("\n[DEBUG] Running multi-objective optimization process...")

    n_obj = 3  # Fixed to 3 objectives

    # ✅ Debugging: Print inputs
    print(f"[DEBUG] Input DataFrame shape: {df.shape}")
    print(f"[DEBUG] n_var: {n_var}, n_obj: {n_obj}")
    print(f"[DEBUG] target_columns: {target_columns} (Expected 3 objectives)")

    # Ensure bounds are numeric
    lower_bounds = [int(b) for b in lower_bounds]
    upper_bounds = [int(b) for b in upper_bounds]

    # ✅ Debugging: Check bounds
    print(f"[DEBUG] lower_bounds: {lower_bounds}")
    print(f"[DEBUG] upper_bounds: {upper_bounds}")

    # Define bounds as tensors
    lower_bounds_tensor = torch.tensor(lower_bounds, **tkwargs)
    upper_bounds_tensor = torch.tensor(upper_bounds, **tkwargs)
    problem_bounds = torch.vstack([lower_bounds_tensor, upper_bounds_tensor])

    # ✅ Debugging: Check bounds tensors
    print(f"[DEBUG] problem_bounds shape: {problem_bounds.shape} (Expected: [2, {n_var}])")

    # Normalize standard bounds for acquisition function
    standard_bounds = torch.zeros(2, n_var, **tkwargs)
    standard_bounds[1] = 1

    # Set random seed
    torch.manual_seed(random_state)

    # Prepare training data
    train_x = torch.tensor(df.iloc[:, :n_var].to_numpy(), **tkwargs)
    train_obj = torch.tensor(df[target_columns].to_numpy(), **tkwargs)

    # ✅ Debugging: Check tensor shapes
    print(f"[DEBUG] train_x shape: {train_x.shape} (Expected: [*, {n_var}])")
    print(f"[DEBUG] train_obj shape: {train_obj.shape} (Expected: [*, {n_obj}])")

    # Normalize inputs and define the model
    train_x_gp = normalize(train_x, problem_bounds)
    models = []

    # Train separate GP models for each objective
    for i in range(n_obj):
        train_y = train_obj[..., i:i + 1]
        models.append(SingleTaskGP(train_x_gp, train_y, outcome_transform=Standardize(m=1)))
        print(f"[DEBUG] GP model {i} trained with train_y shape: {train_y.shape}")

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    # Train Gaussian Process models
    fit_gpytorch_model(mll)
    
    # ✅ Debugging: Check reference_point size
    print(f"[DEBUG] reference_point: {reference_point} (Expected length: 3)")

    ref_point_2 = torch.tensor([float(x) for x in reference_point], **tkwargs)
    print(f"[DEBUG] ref_point_2 shape: {ref_point_2.shape} (Expected: [3])")

    # Define acquisition function
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_2,
        X_baseline=train_x_gp,
        sampler=SobolQMCNormalSampler(torch.Size([512])),
        objective=IdentityMCMultiOutputObjective(outcomes=np.arange(n_obj).tolist()),
        prune_baseline=True,
        cache_pending=True
    )

    # Optimize acquisition function to find the next candidates
    print("[DEBUG] Running acquisition function optimization...")
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

    # ✅ Debugging: Check results
    print(f"[DEBUG] Optimized candidates shape: {df_candidates.shape}")
    print("[DEBUG] Optimized candidates (first 5 rows):")
    print(df_candidates.head())

    return df_candidates
