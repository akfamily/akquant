import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from akquant.ml import PyTorchAdapter, SklearnAdapter
from sklearn.linear_model import LogisticRegression


def verify_sklearn_adapter() -> None:
    """Verify SklearnAdapter functionality."""
    print("\n=== Verifying SklearnAdapter ===")

    # Generate synthetic data
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary classification

    # Initialize adapter
    print("Initializing SklearnAdapter with LogisticRegression...")
    model = LogisticRegression()
    adapter = SklearnAdapter(model)

    # Train
    print("Training...")
    adapter.fit(X, y)

    # Predict
    print("Predicting...")
    preds = adapter.predict(X[:5])
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions: {preds}")

    # Save/Load
    print("Testing Save/Load...")
    path = "temp_sklearn_model.pkl"
    adapter.save(path)

    new_adapter = SklearnAdapter(LogisticRegression())  # Placeholder
    new_adapter.load(path)
    new_preds = new_adapter.predict(X[:5])

    assert np.allclose(preds, new_preds), "Loaded model predictions do not match!"
    print("Save/Load successful.")

    # Cleanup
    if os.path.exists(path):
        os.remove(path)
    print("SklearnAdapter verification passed!")


def verify_pytorch_adapter() -> None:
    """Verify PyTorchAdapter functionality."""
    print("\n=== Verifying PyTorchAdapter ===")

    # Generate synthetic data
    X = np.random.rand(100, 5).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)  # Regression target

    # Define simple network
    class SimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(5, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    # Initialize adapter
    print("Initializing PyTorchAdapter with SimpleNet...")
    network = SimpleNet()
    adapter = PyTorchAdapter(
        network=network,
        criterion=nn.MSELoss(),
        optimizer_cls=optim.Adam,
        lr=0.01,
        epochs=10,
        batch_size=10,
    )

    # Train
    print("Training...")
    adapter.fit(X, y)

    # Predict
    print("Predicting...")
    preds = adapter.predict(X[:5])
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions: {preds}")

    assert isinstance(preds, np.ndarray), "Prediction should be numpy array"

    # Save/Load
    print("Testing Save/Load...")
    path = "temp_torch_model.pth"
    adapter.save(path)

    # Create new adapter with fresh network
    new_network = SimpleNet()
    new_adapter = PyTorchAdapter(
        network=new_network, criterion=nn.MSELoss(), optimizer_cls=optim.Adam
    )
    new_adapter.load(path)
    new_preds = new_adapter.predict(X[:5])

    assert np.allclose(preds, new_preds), "Loaded model predictions do not match!"
    print("Save/Load successful.")

    # Cleanup
    if os.path.exists(path):
        os.remove(path)
    print("PyTorchAdapter verification passed!")


if __name__ == "__main__":
    try:
        verify_sklearn_adapter()
        verify_pytorch_adapter()
        print("\nAll verifications passed successfully!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
