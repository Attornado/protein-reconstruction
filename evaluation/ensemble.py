from typing import List, final, Union, Any, Optional
import torch
from torch.nn.functional import softmax
from models.classification.classifiers import GraphClassifier


VOTING: final = "voting"
SOFTMAX_MEAN: final = "softmax_mean"
_ENSEMBLE_MODES: final = frozenset([SOFTMAX_MEAN, VOTING])


class EnsembleGraphClassifier(torch.nn.Module):
    def __init__(self,
                 models: List[GraphClassifier],
                 dim_features: int,
                 dim_target: int,
                 ensemble_mode: str = SOFTMAX_MEAN,
                 weights: Optional[List[float]] = None,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        if ensemble_mode not in _ENSEMBLE_MODES:
            raise ValueError(f"ensemble_mode must be in {_ENSEMBLE_MODES}. {ensemble_mode} given.")
        if weights is not None and len(weights) != len(models):
            raise ValueError(f"weights and models must have the same length. {len(weights)} and {len(models)} given.")

        self.__dim_features: int = dim_features
        self.__dim_target: int = dim_target
        self.__ensemble_mode: str = ensemble_mode
        self.__device: torch.device = device
        self._models = torch.nn.ModuleList(models)

        # Compute weights, 1 for each model if not given
        weights = torch.tensor(weights) if weights is not None else torch.tensor([1.0 for _ in range(0, len(models))])
        weights = weights
        self.__weights = weights

    @property
    def ensemble_mode(self) -> str:
        return self.__ensemble_mode

    @ensemble_mode.setter
    def ensemble_mode(self, ensemble_mode: str):
        self.__ensemble_mode = ensemble_mode

    @property
    def weights(self) -> torch.Tensor:
        return self.__weights

    @property
    def dim_features(self) -> int:
        return self.__dim_features

    @property
    def dim_target(self) -> int:
        return self.__dim_target

    @property
    def device(self) -> torch.device:
        return self.__device

    def forward(self, *args, **kwargs) -> List[torch.Tensor]:

        outputs = []
        # For each model in the ensemble
        for model in self._models:

            # Get model prediction and store it
            model = model.to(self.device)
            outputs.append(model(*args, **kwargs))

            # Free memory
            del model
            torch.cuda.empty_cache()

        return outputs

    def ensemble(self, return_probs: bool = False, *args, **kwargs) -> torch.Tensor:
        """
        Gets the ensemble predictions.

        :param return_probs: whether to return ensemble probabilities alongside predicted classes. Used only when using
            softmax mean ensemble.

        :return: the ensemble predictions.
        """
        # Get list with all model predictions as logits vector with shape (B, N_CLASSES)
        logits_ensemble: List[torch.Tensor] = self(*args, **kwargs)

        # Apply softmax to each logits vector, adding a dim to the softmax vector to obtain shape (B, 1, N_CLASSES)
        preds_ensemble = [softmax(logits, dim=-1).unsqueeze(dim=1) for logits in logits_ensemble]

        # Concat the logits in the same batch, obtaining tensor with shape (B, N_MODELS, N_CLASSES)
        preds_ensemble = torch.concat(preds_ensemble, dim=1)

        # If averaged softmax vectors are required
        if self.ensemble_mode == SOFTMAX_MEAN:
            # Compute weighted average of the softmax, multiplying the weights with the Einstein's notation
            # Here we have two tensors, with shape (B, N_MODELS, N_CLASSES) and (N_MODELS,), and the following means
            # that we are element-wise multiplying the 1st tensor alongside the dimension 2 of the 2nd tensor
            preds_ensemble = torch.einsum("j,ijk -> ijk", self.weights, preds_ensemble)

            # Sum the tensor with shape (B, N_MODELS, N_CLASSES) alongside dimension 1, obtaining shape (B, N_CLASSES)
            preds_ensemble = torch.sum(preds_ensemble, dim=1)

            # Divide by the sum of the weights to obtain the weighted average of the softmax vectors
            preds_ensemble = preds_ensemble / torch.sum(self.weights)

            if return_probs:
                return preds_ensemble

        # Get predictions from probabilities
        y_pred_ensemble = torch.argmax(preds_ensemble, dim=-1)

        # If hard/soft voting is required
        if self.ensemble_mode == VOTING:

            # For each ensemble prediction vector
            true_preds = []
            for b in range(0, y_pred_ensemble.shape[0]):

                # Count weighted frequencies of each class and get the corresponding prediction with argmax
                weighted_frequencies = torch.bincount(y_pred_ensemble[b], weights=self.weights)
                predicted_index = torch.argmax(weighted_frequencies)
                predicted = y_pred_ensemble[b][predicted_index]
                true_preds.append(predicted)

            # Get final predictions
            preds_ensemble = torch.tensor(true_preds)

        return preds_ensemble


