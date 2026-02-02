"""Quick test of the trained model."""

import torch
from src.models import UrbanSceneCNN
from pathlib import Path

# Load the best model
model = UrbanSceneCNN(num_classes=5)
checkpoint_path = Path("checkpoints/best_model.pth")

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded successfully!")
    print(model.summary())
    
    # Test inference
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        output = model(x)
    print(f"\n✓ Model inference test passed!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits):\n{output}")
else:
    print("Model checkpoint not found!")
