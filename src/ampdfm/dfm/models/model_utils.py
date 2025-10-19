import torch

from ampdfm.dfm.flow_matching.path import MixtureDiscreteProbPath
from ampdfm.dfm.flow_matching.path.scheduler import PolynomialConvexScheduler
from ampdfm.dfm.flow_matching.solver import MixtureDiscreteEulerSolver
from ampdfm.dfm.flow_matching.utils import ModelWrapper

from ampdfm.dfm.models.peptide_models import CNNModelPep


def load_solver(checkpoint_path, vocab_size, device, embed_dim=1024, hidden_dim=512):
    """Load a trained CNNModelPep checkpoint and wrap it in a MixtureDiscreteEulerSolver.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint.
        vocab_size: Vocabulary size (typically 24 for ampdfm).
        device: Device to load the model on (e.g., 'cuda:0' or 'cpu').
        embed_dim: Token embedding dimension (default 1024 for unconditional ampdfm).
        hidden_dim: Hidden channel dimension (default 512 for unconditional ampdfm).
    
    Returns:
        MixtureDiscreteEulerSolver: Configured solver for discrete sampling.
    """
    probability_denoiser = CNNModelPep(alphabet_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    probability_denoiser.load_state_dict(torch.load(checkpoint_path, map_location=device))
    probability_denoiser.eval()
    for param in probability_denoiser.parameters():
        param.requires_grad = False

    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)

    class WrappedModel(ModelWrapper):
        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
            return torch.softmax(self.model(x, t), dim=-1)

    wrapped_probability_denoiser = WrappedModel(probability_denoiser)
    solver = MixtureDiscreteEulerSolver(model=wrapped_probability_denoiser, path=path, vocabulary_size=vocab_size)

    return solver
