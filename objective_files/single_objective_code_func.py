import pandas as pd
import torch
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch import fit_gpytorch_model

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def run_optimization(
    df: pd.DataFrame,
    lower_bounds: list,
    upper_bounds: list,
    n_var: int,
    n_obj: int,
    batch_size: int,
    target_column: str = "target",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Runs the optimization process using the provided data and parameters.
    
    Args:
        df (pd.DataFrame): Input dataset with feature columns and target column(s).
        lower_bounds (list): Lower bounds for the optimization variables.
        upper_bounds (list): Upper bounds for the optimization variables.
        n_var (int): Number of variables (features) to consider.
        n_obj (int): Number of objectives (target columns).
        batch_size (int): Batch size for the optimization.
        random_state (int): Seed for reproducibility.
    
    Returns:
        pd.DataFrame: DataFrame containing the optimized candidates.
    """
    # Debug: print the starting of the optimization process
    print("Running optimization process...")
    
    # Debug: print the initial state of inputs
    print("DataFrame head:\n", df.head())
    print("Lower bounds:\n", lower_bounds)
    print("Upper bounds:\n", upper_bounds)
    print("n_var:", n_var)
    print("n_obj:", n_obj)
    print("batch_size:", batch_size)
    print("target_column:", target_column)
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
    train_obj = torch.tensor(df[[target_column]].to_numpy(), **tkwargs)

    # Debug: print training data shapes
    print(f"train_x shape: {train_x.shape}")
    print(f"train_obj shape: {train_obj.shape}")

    # Normalize inputs and define the model
    train_x_gp = normalize(train_x, problem_bounds)
    train_y = train_obj
    model = SingleTaskGP(train_x_gp, train_y)

    # Debug: print model state
    print("Model initialized")

    # Train Gaussian Process model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Debug: print training status
    print("Model trained")

    # Define acquisition function
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]), resample=False, seed=0)
    acqf = qLogExpectedImprovement(model, best_f=torch.max(model.train_targets), sampler=sampler)

    # Debug: print acquisition function status
    print("Acquisition function initialized")

    # Optimize acquisition function to find the next candidates
    next_x, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=batch_size,
        num_restarts=50,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # Debug: print next_x shape
    print(f"Next_x shape: {next_x.shape}")

    # Unnormalize and return the optimized candidates
    new_x = unnormalize(next_x.detach(), bounds=problem_bounds)
    new_x = new_x.cpu().numpy()
    df_candidates = pd.DataFrame(new_x, columns=df.columns[:n_var])
    df_candidates = df_candidates.round(2)

    # Debug: print first few rows of the resulting dataframe
    print("Optimized candidates:")

    return df_candidates
