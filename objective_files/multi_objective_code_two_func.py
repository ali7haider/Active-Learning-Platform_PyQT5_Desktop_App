import pandas as pd
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.exceptions import BadInitialCandidatesWarning
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_model

import warnings

# Ensure warnings are suppressed
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)

def run_multi_objective_optimization_two(
    df: pd.DataFrame,
    lower_bounds: list,
    upper_bounds: list,
    n_var: int,
    n_obj: int,
    batch_size: int,
    ref_point: list = [0, 0],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Runs the multi-objective optimization process using the provided data and parameters.
    
    Args:
        df (pd.DataFrame): Input dataset with feature columns and target column(s).
        lower_bounds (list): Lower bounds for the optimization variables.
        upper_bounds (list): Upper bounds for the optimization variables.
        n_var (int): Number of variables (features) to consider.
        n_obj (int): Number of objectives (target columns).
        batch_size (int): Batch size for the optimization.
        ref_point (list): Reference point for hypervolume calculation.
        random_state (int): Seed for reproducibility.
    
    Returns:
        pd.DataFrame: DataFrame containing the optimized candidates.
    """
    # Debug: print the starting of the optimization process
    print("Running multi-objective optimization process...")
    
    # Debug: print the initial state of inputs
    print("DataFrame head:\n", df.head())
    print("Lower bounds:\n", lower_bounds)
    print("Upper bounds:\n", upper_bounds)
    print("n_var:", n_var)
    print("n_obj:", n_obj)
    print("batch_size:", batch_size)
    print("ref_point:", ref_point)
    print("random_state:", random_state)

    # Ensure bounds are numeric
    lower_bounds = [float(b) for b in lower_bounds]  # Convert lower bounds to float
    upper_bounds = [float(b) for b in upper_bounds]  # Convert upper bounds to float
    
    # Define tensor properties
    tkwargs = {"dtype": torch.double, "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}
    
    # Define bounds as tensors
    lower_bounds_tensor = torch.tensor(lower_bounds, **tkwargs)
    upper_bounds_tensor = torch.tensor(upper_bounds, **tkwargs)
    problem_bounds = torch.vstack([lower_bounds_tensor, upper_bounds_tensor])

    # Debug: print tensor bounds
    print(f"Lower bounds tensor:\n{lower_bounds_tensor}")
    print(f"Upper bounds tensor:\n{upper_bounds_tensor}")

    # Normalize standard bounds for acquisition function
    standard_bounds = torch.zeros(2, n_var, **tkwargs)
    standard_bounds[1] = 1

    # Debug: print standard bounds
    print(f"Standard bounds:\n{standard_bounds}")

    # Set random seed
    torch.manual_seed(random_state)

    # Prepare training data
    train_x = torch.tensor(df.iloc[:, :n_var].to_numpy(), **tkwargs)
    train_obj = torch.tensor(df.iloc[:, n_var:n_var + n_obj].to_numpy(), **tkwargs)

    # Debug: print training data shapes
    print(f"train_x shape: {train_x.shape}")
    print(f"train_obj shape: {train_obj.shape}")

    # Normalize inputs and define the model
    train_x_gp = normalize(train_x, problem_bounds)
    models = []

    # Create and train separate models for each objective
    for i in range(train_obj.shape[-1]): 
        train_y = train_obj[..., i:i + 1]
        models.append(SingleTaskGP(train_x_gp, train_y, outcome_transform=Standardize(m=1)))
    model = ModelListGP(*models)

    # Debug: print model state
    print("Model initialized")

    # Train Gaussian Process model
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Debug: print training status
    print("Model trained")

    # Define acquisition function
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=torch.tensor(ref_point, dtype=torch.double, **tkwargs),
        X_baseline=train_x_gp,
        sampler=SobolQMCNormalSampler(torch.Size([512])),
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(n_obj))),
        prune_baseline=True,
        cache_pending=True
    )

    # Debug: print acquisition function status
    print("Acquisition function initialized")

    # Optimize acquisition function to find the next candidates
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        q=batch_size,
        bounds=standard_bounds,
        num_restarts=10,
        raw_samples=128,
        options={"batch_limit": 5, "maxiter": 200}
    )

    # Debug: print candidates shape
    print(f"Optimized candidates shape: {candidates.shape}")

    # Unnormalize and return the optimized candidates
    new_x = unnormalize(candidates.detach(), bounds=problem_bounds)
    new_x = new_x.cpu().numpy()
    df_candidates = pd.DataFrame(new_x, columns=df.columns[:n_var])
    df_candidates = df_candidates.round(2)

    # Debug: print first few rows of the resulting dataframe
    print("Optimized candidates:\n", df_candidates.head())

    return df_candidates
