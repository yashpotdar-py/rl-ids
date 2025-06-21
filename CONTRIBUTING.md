# Contributing to RL-IDS

We welcome contributions from the community! This document provides guidelines for contributing to the RL-IDS Adaptive System.

## 🚀 Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/ -v --cov=rl_ids`
5. Update documentation if needed
6. Submit a pull request

## 📋 Development Setup

### Prerequisites
- Python 3.8+
- Git
- Docker (optional)

### Local Development
```bash
# Clone your fork
git clone https://github.com/your-username/rl-ids.git
cd rl-ids

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=rl_ids --cov-report=html

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
```

### Writing Tests
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Follow the naming convention: `test_*.py`
- Use descriptive test names: `test_dqn_agent_training_convergence`

## 📝 Code Style

We use several tools to maintain code quality:

```bash
# Format code
black rl_ids/ tests/
isort rl_ids/ tests/

# Type checking
mypy rl_ids/

# Linting
flake8 rl_ids/ tests/
```

### Style Guidelines
- Follow PEP 8 for Python code style
- Use type hints for all public functions
- Write docstrings for modules, classes, and functions
- Keep line length under 88 characters (Black default)

## 📚 Documentation

### Building Documentation
```bash
# Install documentation dependencies
pip install mkdocs-material

# Serve locally with hot reload
cd docs/
mkdocs serve

# Build static documentation
mkdocs build
```

### Documentation Guidelines
- Update relevant documentation for any API changes
- Add new tutorials for significant features
- Keep examples up-to-date and tested
- Use clear, concise language

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - Operating system
   - Package versions (`pip freeze`)

2. **Reproduction Steps**
   - Minimal code example
   - Expected vs. actual behavior
   - Error messages and stack traces

3. **Additional Context**
   - Configuration files
   - Log outputs
   - Screenshots (if applicable)

## ✨ Feature Requests

For new features:

1. **Check existing issues** to avoid duplicates
2. **Describe the problem** the feature would solve
3. **Propose a solution** or implementation approach
4. **Consider backwards compatibility**

## 📦 Pull Request Process

### Before Submitting
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Code is formatted: `black rl_ids/ tests/`
- [ ] Imports are sorted: `isort rl_ids/ tests/`
- [ ] Type checks pass: `mypy rl_ids/`
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] New tests added
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## 🏗️ Architecture Guidelines

### Code Organization
```
rl_ids/
├── agents/          # RL agents (DQN, etc.)
├── environments/    # Gymnasium environments
├── modeling/        # Training and evaluation
├── make_dataset.py  # Data processing
├── plots.py         # Visualization
└── config.py        # Configuration

api/
├── main.py          # FastAPI application
├── models/          # Pydantic models
├── routes/          # API endpoints
└── services/        # Business logic
```

### Design Principles
- **Modularity**: Keep components loosely coupled
- **Testability**: Design for easy testing
- **Configuration**: Make components configurable
- **Documentation**: Document public APIs

## 🎯 Contribution Areas

### High Priority
- Performance optimizations
- Additional RL algorithms
- Enhanced monitoring and alerting
- Security improvements

### Medium Priority
- Additional dataset support
- Visualization enhancements
- API feature additions
- Documentation improvements

### Welcome Contributions
- Bug fixes and improvements
- Test coverage expansion
- Example scripts and tutorials
- Typo fixes and documentation clarity

## 📞 Getting Help

- **Documentation**: Check [docs/](docs/) for comprehensive guides
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for security issues

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Documentation credits

Thank you for contributing to RL-IDS! 🚀