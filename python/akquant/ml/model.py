from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

# Define unified data input type
DataType = Union[np.ndarray, pd.DataFrame]


@dataclass
class ValidationConfig:
    """Configuration for model validation."""

    method: Literal["walk_forward"] = "walk_forward"
    train_window: Union[str, int] = "1y"
    test_window: Union[str, int] = (
        "3m"  # Not strictly used in rolling execution, but useful for evaluation
    )
    rolling_step: Union[str, int] = "3m"
    frequency: str = "1d"
    verbose: bool = False


class QuantModel(ABC):
    """
    Abstract base class for all quantitative models.

    The strategy layer only interacts with this class, not directly with sklearn or
    torch.
    """

    def __init__(self) -> None:
        """Initialize the model."""
        self.validation_config: Optional[ValidationConfig] = None

    def set_validation(
        self,
        method: Literal["walk_forward"] = "walk_forward",
        train_window: Union[str, int] = "1y",
        test_window: Union[str, int] = "3m",
        rolling_step: Union[str, int] = "3m",
        frequency: str = "1d",
        verbose: bool = False,
    ) -> None:
        """
        Configure validation method (e.g., Walk-forward).

        :param method: Validation method (currently only 'walk_forward').
        :param train_window: Training data duration (e.g., '1y', '250d') or bar count.
        :param test_window: Testing/Prediction duration (e.g., '3m') or bar count.
        :param rolling_step: How often to retrain (e.g., '3m') or bar count.
        :param frequency: Data frequency ('1d', '1h', '1m') used for parsing time
            strings.
        :param verbose: Whether to print training logs (default False).
        """
        self.validation_config = ValidationConfig(
            method=method,
            train_window=train_window,
            test_window=test_window,
            rolling_step=rolling_step,
            frequency=frequency,
            verbose=verbose,
        )

    @abstractmethod
    def fit(self, X: DataType, y: DataType) -> None:
        """
        Train the model.

        Args:
            X: Training features
            y: Training labels
        """
        pass

    @abstractmethod
    def predict(self, X: DataType) -> np.ndarray:
        """
        Predict using the model.

        Args:
            X: Input features

        Returns:
            np.ndarray: Prediction results (numpy array)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to the specified path."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from the specified path."""
        pass


class SklearnAdapter(QuantModel):
    """Adapter for Scikit-learn style models."""

    def __init__(self, estimator: Any):
        """
        Initialize the adapter.

        Args:
            estimator: A sklearn-style estimator instance (e.g., XGBClassifier,
                LGBMRegressor)
        """
        super().__init__()
        self.model = estimator

    def fit(self, X: DataType, y: DataType) -> None:
        """Train the sklearn model."""
        if self.validation_config and self.validation_config.verbose:
            print(f"Training Sklearn Model: {type(self.model).__name__}")

        # Convert DataFrame to numpy if necessary, or let sklearn handle it
        self.model.fit(X, y)

    def predict(self, X: DataType) -> np.ndarray:
        """Predict using the sklearn model."""
        # For classification, we usually care about the probability of class 1
        if hasattr(self.model, "predict_proba"):
            # Return probability of class 1
            # Note: This assumes binary classification. For multi-class, this might
            # need adjustment.
            proba = self.model.predict_proba(X)
            if proba.shape[1] > 1:
                return proba[:, 1]  # type: ignore
            return proba  # type: ignore
        else:
            return self.model.predict(X)  # type: ignore

    def save(self, path: str) -> None:
        """Save the sklearn model using joblib."""
        import joblib  # type: ignore

        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """Load the sklearn model using joblib."""
        import joblib  # type: ignore

        self.model = joblib.load(path)


class PyTorchAdapter(QuantModel):
    """Adapter for PyTorch models."""

    def __init__(
        self,
        network: Any,
        criterion: Any,
        optimizer_cls: Any,
        lr: float = 0.001,
        epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize the PyTorch adapter.

        Args:
            network: PyTorch neural network module (nn.Module)
            criterion: Loss function (nn.Module)
            optimizer_cls: Optimizer class (torch.optim.Optimizer)
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            device: Device to run on ('cpu' or 'cuda')
        """
        super().__init__()
        import torch

        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer_cls(self.network.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X: DataType, y: DataType) -> None:
        """Train the PyTorch model."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # 1. Data conversion: Numpy/Pandas -> Tensor
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = (
            y.values if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series) else y
        )

        X_tensor = torch.tensor(X_array, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_array, dtype=torch.float32).to(self.device)

        # 2. Wrap in DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 3. Standard training loop
        self.network.train()

        verbose = False
        if self.validation_config and self.validation_config.verbose:
            verbose = True

        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.network(batch_X)

                # Note: Adjust loss calculation dimensions based on task
                # (regression/classification)
                # Squeeze last dim if it's 1 (e.g. (N, 1) -> (N)) to match batch_y
                if outputs.dim() > 1 and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)

                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            if verbose:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")

    def predict(self, X: DataType) -> np.ndarray:
        """
        Predict using the PyTorch model.

        Note:
            This returns the raw output from the network (logits or probabilities
            depending on the network's last layer). User should handle any necessary
            activations (e.g. sigmoid, softmax) in the network definition or
            post-processing.
        """
        import torch

        self.network.eval()
        with torch.no_grad():
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            X_tensor = torch.tensor(X_array, dtype=torch.float32).to(self.device)
            outputs = self.network(X_tensor)
            # Convert back to Numpy for strategy layer
            return outputs.cpu().numpy().flatten()  # type: ignore

    def save(self, path: str) -> None:
        """Save the PyTorch model state dict."""
        import torch

        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the PyTorch model state dict."""
        import torch

        self.network.load_state_dict(torch.load(path))
