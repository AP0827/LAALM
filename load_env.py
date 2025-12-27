"""Load environment variables from .env file."""

import os
from pathlib import Path


def load_env_file(env_path=".env"):
    """Load environment variables from .env file."""
    env_file = Path(env_path)
    
    if not env_file.exists():
        return False
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if value.strip():
                    os.environ[key.strip()] = value.strip()
    
    return True


if __name__ == '__main__':
    if load_env_file():
        print("✓ Environment variables loaded")
    else:
        print("✗ .env file not found")
