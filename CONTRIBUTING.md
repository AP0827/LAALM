# Contributing to LAALM

Thank you for your interest in contributing to LAALM! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Process](#development-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, background, or identity.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Harassment, trolling, or derogatory comments
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- Basic understanding of multi-modal AI systems
- Familiarity with PyTorch, FastAPI, or React (depending on contribution area)

### Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork:**
```bash
git clone https://github.com/YOUR_USERNAME/LAALM.git
cd LAALM
```

3. **Add upstream remote:**
```bash
git remote add upstream https://github.com/AP0827/LAALM.git
```

4. **Create virtual environment:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

5. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy  # Development tools
```

6. **Configure API keys:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

## How to Contribute

### Areas for Contribution

We welcome contributions in the following areas:

**Code:**
- Bug fixes
- New features
- Performance improvements
- Code refactoring

**Documentation:**
- Improving existing docs
- Adding examples
- Translating documentation
- Creating tutorials

**Testing:**
- Writing unit tests
- Integration tests
- Performance benchmarks

**Design:**
- UI/UX improvements
- Visual assets
- Accessibility enhancements

## Development Process

### 1. Find or Create an Issue

- Check [existing issues](https://github.com/AP0827/LAALM/issues)
- Create a new issue if needed
- Comment on the issue to claim it

### 2. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions

### 3. Make Changes

- Write clean, readable code
- Follow the code style guide
- Add tests for new features
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run tests
pytest

# Check code style
black .
flake8 .

# Type checking
mypy pipeline.py
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

**Commit message format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Build/tooling changes

**Example:**
```
feat: add MediaPipe face detector support

- Implement MediaPipe detector class
- Add configuration option for detector selection
- Update documentation with usage examples

Closes #123
```

### 6. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 7. Create Pull Request

- Go to your fork on GitHub
- Click "New Pull Request"
- Fill out the PR template
- Link related issues

## Code Style

### Python

Follow [PEP 8](https://pep8.org/) with these specifics:

**Formatting:**
- Use [Black](https://black.readthedocs.io/) for code formatting
- Line length: 88 characters
- Use double quotes for strings

**Type Hints:**
```python
from typing import Dict, List, Optional

def process_video(
    video_path: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process video file."""
    pass
```

**Docstrings:**
```python
def complex_function(param1: str, param2: int) -> bool:
    """One-line summary.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
    """
    pass
```

### JavaScript/React

**Formatting:**
- Use Prettier for formatting
- 2 spaces for indentation
- Semicolons required

**Component Structure:**
```javascript
import React, { useState } from 'react';

export default function MyComponent({ prop1, prop2 }) {
  const [state, setState] = useState(null);
  
  const handleEvent = () => {
    // Handler logic
  };
  
  return (
    <div className="container">
      {/* JSX */}
    </div>
  );
}
```

## Testing

### Writing Tests

**Unit Tests:**
```python
# tests/test_module.py
import pytest
from module import function_to_test

def test_function_basic():
    """Test basic functionality."""
    result = function_to_test("input")
    assert result == "expected"

def test_function_edge_case():
    """Test edge case."""
    with pytest.raises(ValueError):
        function_to_test(None)
```

**Integration Tests:**
```python
def test_pipeline_integration():
    """Test complete pipeline."""
    result = run_mvp(
        video_file="samples/video/test.mpg",
        audio_file="samples/audio/test.wav"
    )
    assert "final_transcript" in result
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pipeline.py

# Run with coverage
pytest --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include regression tests

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No new warnings

## Related Issues
Closes #123
```

### Review Process

1. **Automated Checks:** CI/CD runs tests and linting
2. **Code Review:** Maintainers review your code
3. **Feedback:** Address review comments
4. **Approval:** At least one maintainer approval required
5. **Merge:** Maintainer merges your PR

### After Merge

- Delete your feature branch
- Update your fork:
```bash
git checkout main
git pull upstream main
git push origin main
```

## Issue Reporting

### Bug Reports

**Template:**
```markdown
## Bug Description
Clear description of the bug

## To Reproduce
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- LAALM version: [e.g., 1.0.0]

## Additional Context
Screenshots, logs, etc.
```

### Feature Requests

**Template:**
```markdown
## Feature Description
Clear description of the feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should this work?

## Alternatives Considered
Other approaches you've thought about

## Additional Context
Any other relevant information
```

### Questions

For questions:
- Check existing documentation
- Search closed issues
- Ask in discussions (if available)
- Create an issue with `question` label

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Contact

- **Project Maintainers:**
  - Asish Kumar Yeleti: asishkumary.is23@rvce.edu.in
  - Aayush Pandey: aayushpandey.is23@rvce.edu.in

- **Institution:** R V College of Engineering

## Additional Resources

- [Development Guide](docs/DEVELOPMENT.md)
- [Architecture Documentation](docs/ARCHITECTURE.md)
- [API Documentation](docs/API.md)

---

**Thank you for contributing to LAALM! 🎉**
