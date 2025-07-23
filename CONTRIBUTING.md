# Contributing to MAHT-Net

Thank you for your interest in contributing to MAHT-Net! This project aims to advance medical AI for cephalometric landmark detection with clinical-grade precision.

## Code of Conduct

This project follows a professional and inclusive code of conduct. Please be respectful, constructive, and collaborative in all interactions.

## How to Contribute

### Reporting Issues

1. **Search existing issues** to avoid duplicates
2. **Use the issue template** when available
3. **Provide detailed information**:
   - Environment details (OS, Python version, dependencies)
   - Steps to reproduce the issue
   - Expected vs. actual behavior
   - Error messages and logs
   - Sample data or minimal reproducible example

### Suggesting Enhancements

1. **Check existing feature requests** to avoid duplicates
2. **Describe the enhancement** clearly:
   - Use case and motivation
   - Proposed solution
   - Alternative approaches considered
   - Potential impact on existing functionality

### Contributing Code

#### Development Setup

```bash
# Clone the repository
git clone https://github.com/mohamednourdine/maht-net.git
cd maht-net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### Development Workflow

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the project conventions:
   - Follow existing code style (Black, isort, flake8)
   - Add docstrings for new functions and classes
   - Include type hints where appropriate
   - Write tests for new functionality

3. **Test your changes**:
   ```bash
   # Run tests
   pytest tests/

   # Run linting
   flake8 src/
   black --check src/
   isort --check-only src/

   # Run type checking
   mypy src/
   ```

4. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "feat: add multi-scale attention mechanism"
   ```

5. **Push to your fork** and submit a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

#### Pull Request Guidelines

- **Use descriptive titles** and provide detailed descriptions
- **Reference related issues** using keywords (fixes #123, closes #456)
- **Include tests** for new features and bug fixes
- **Update documentation** as needed
- **Ensure CI passes** before requesting review
- **Keep PRs focused** - one feature or fix per PR

### Code Style Guidelines

#### Python Code Style

- **Follow PEP 8** with Black formatting (line length: 88)
- **Use type hints** for function signatures
- **Write docstrings** using Google style:
  ```python
  def process_image(image: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
      """Process cephalometric image with landmark annotations.
      
      Args:
          image: Input grayscale image array of shape (H, W)
          landmarks: Landmark coordinates of shape (N, 2)
          
      Returns:
          Dictionary containing processed image and metadata
          
      Raises:
          ValueError: If image dimensions are invalid
      """
  ```

#### Commit Message Format

Use conventional commits format:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` code style changes (formatting, etc.)
- `refactor:` code refactoring
- `test:` adding or updating tests
- `chore:` maintenance tasks

Examples:
```
feat: implement attention-gated decoder module
fix: resolve memory leak in data loader
docs: update training strategy documentation
test: add unit tests for heatmap generation
```

### Testing Guidelines

#### Test Organization

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Benchmark critical operations

#### Test Structure

```python
# tests/test_models/test_maht_net.py
import pytest
import torch
from src.models.maht_net import MAHTNet

class TestMAHTNet:
    """Test suite for MAHT-Net model."""
    
    @pytest.fixture
    def model(self):
        """Create test model instance."""
        return MAHTNet(num_landmarks=7, image_size=512)
    
    def test_forward_pass(self, model):
        """Test model forward pass with valid input."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 512, 512)
        
        output = model(input_tensor)
        
        assert output['heatmaps'].shape == (batch_size, 7, 128, 128)
        assert output['coordinates'].shape == (batch_size, 7, 2)
```

### Documentation Guidelines

#### Code Documentation

- **Module docstrings**: Describe module purpose and usage
- **Class docstrings**: Explain class functionality and attributes
- **Function docstrings**: Document parameters, returns, and exceptions
- **Inline comments**: Explain complex logic and algorithms

#### Project Documentation

- **Keep documentation updated** with code changes
- **Use clear, concise language** accessible to researchers and developers
- **Include examples** and code snippets where helpful
- **Maintain consistency** with existing documentation style

### Medical Data and Privacy

#### Data Handling

- **Never commit patient data** or personal information
- **Use synthetic or anonymized data** for examples
- **Follow medical data privacy regulations** (HIPAA, GDPR)
- **Document data sources** and usage permissions

#### Ethical Considerations

- **Consider bias** in datasets and algorithms
- **Document limitations** and potential failure modes
- **Include uncertainty quantification** for clinical applications
- **Provide clear disclaimers** about medical use

### Release Process

1. **Version bumping**: Follow semantic versioning (MAJOR.MINOR.PATCH)
2. **Changelog update**: Document changes in CHANGELOG.md
3. **Tag creation**: Create annotated git tags for releases
4. **Documentation update**: Ensure docs reflect new version

### Getting Help

- **Join discussions** in GitHub Discussions
- **Ask questions** in issues with the "question" label
- **Review documentation** in the `/documentation` directory
- **Check existing issues** and pull requests

### Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Academic publications (for research contributions)

Thank you for contributing to advancing medical AI research and improving healthcare outcomes! 🚀
