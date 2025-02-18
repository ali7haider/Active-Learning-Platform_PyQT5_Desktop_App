import pandas as pd
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.exceptions import BadInitialCandidatesWarning
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
import sys
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
import warnings

# Ensure warnings are suppressed
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def run_multi_objective_optimization_two(
    df: pd.DataFrame,
    lower_bounds: list,
    upper_bounds: list,
    n_var: int,
    target_columns: list,
    batch_size: int,
    reference_point: list = [0, 0],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Runs the multi-objective optimization process using the provided data and parameters.
    
    Args:
        df (pd.DataFrame): Input dataset with feature columns and target column(s).
        lower_bounds (list): Lower bounds for the optimization variables.
        upper_bounds (list): Upper bounds for the optimization variables.
        n_var (int): Number of variables (features) to consider.
        batch_size (int): Batch size for the optimization.
        ref_point (list): Reference point for hypervolume calculation.
        random_state (int): Seed for reproducibility.
    
    Returns:
        pd.DataFrame: DataFrame containing the optimized candidates.
    """
    print("Starting Multi-Objective Optimization Process...")
    print("Preparing data and setting up models...")

    # Define tensor properties
    tkwargs = {"dtype": torch.double, "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}
    
    # Define bounds as tensors
    print("Defining bounds for the optimization variables...")
    lower_bounds_tensor = torch.tensor(lower_bounds, **tkwargs)
    upper_bounds_tensor = torch.tensor(upper_bounds, **tkwargs)
    problem_bounds = torch.vstack([lower_bounds_tensor, upper_bounds_tensor])

    # Normalize standard bounds for acquisition function
    standard_bounds = torch.zeros(2, n_var, **tkwargs)
    standard_bounds[1] = 1

    # Set random seed
    torch.manual_seed(random_state)
    print("Random seed set for reproducibility.")

    # Prepare training data
    print("Loading input data for training...")
    train_x = torch.tensor(df.iloc[:, :n_var].to_numpy(), **tkwargs)
    train_obj = torch.tensor(df[target_columns].to_numpy(), **tkwargs)

    # Normalize inputs and define the model
    train_x_gp = normalize(train_x, problem_bounds)
    models = []

    print("Initializing Gaussian Process (GP) models for each objective...")
    # Create and train separate models for each objective
    for i in range(train_obj.shape[-1]): 
        train_y = train_obj[..., i:i + 1]
        models.append(SingleTaskGP(train_x_gp, train_y, outcome_transform=Standardize(m=1)))
    model = ModelListGP(*models)

    print("Model successfully initialized.")

    # Train Gaussian Process model
    print("Training the GP model...")
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    print("Model training completed.")

    ref_point_2 = torch.tensor([float(x) for x in reference_point], **tkwargs)

    print("Initializing acquisition function for optimization...")
    # Define acquisition function
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_2,
        X_baseline=train_x_gp,
        sampler=SobolQMCNormalSampler(torch.Size([512])),
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(2))),
        prune_baseline=True,
        cache_pending=True
    )

    print("Acquisition function successfully initialized.")

    print("Running optimization to find the best candidates...")
    # Optimize acquisition function to find the next candidates
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        q=batch_size,
        bounds=standard_bounds,
        num_restarts=10,
        raw_samples=128,
        options={"batch_limit": 5, "maxiter": 200}
    )

    print("Optimization completed. Extracting results...")

    # Unnormalize and return the optimized candidates
    new_x = unnormalize(candidates.detach(), bounds=problem_bounds)
    new_x = new_x.cpu().numpy()
    df_candidates = pd.DataFrame(new_x, columns=df.columns[:n_var])
    df_candidates = df_candidates.round(2)

    print("Optimized candidates:\n", df_candidates.head())

    print("Multi-objective optimization completed successfully!")
    return df_candidates

