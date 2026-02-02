"""Root pytest configuration - loaded before test collection."""

import sys
from pathlib import Path

# Add the project root to sys.path BEFORE any test imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
