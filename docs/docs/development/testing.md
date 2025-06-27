# Testing Guide

This guide covers the testing strategy, setup, and execution for the RL-IDS project.

## Testing Overview

RL-IDS uses a comprehensive testing approach that includes unit tests, integration tests, and performance tests to ensure reliability and correctness.

## Test Structure

```
tests/
├── test_api.py          # API endpoint tests
├── conftest.py          # Test configuration and fixtures (if exists)
├── unit/                # Unit tests (if exists)
├── integration/         # Integration tests (if exists)
└── performance/         # Performance tests (if exists)
```

## Current Test Suite

### API Tests (`tests/test_api.py`)

The main test suite focuses on API functionality and includes:

#### Test Classes and Methods

**TestRLIDSAPI Class**
- `test_root_endpoint()`: Tests the main API endpoint
- `test_health_check()`: Verifies health check functionality  
- `test_model_info()`: Tests model information endpoint
- `test_single_prediction()`: Tests individual prediction requests
- Additional prediction and batch processing tests

#### Key Test Areas

1. **Endpoint Validation**
   - Response status codes
   - Response data structure
   - Required fields presence

2. **Model Integration**
   - Model loading verification
   - Prediction accuracy
   - Input validation

3. **Error Handling**
   - Invalid input handling
   - Exception responses
   - Graceful degradation

## Running Tests

### Prerequisites

Ensure you have the test dependencies installed:

```bash
# Install test dependencies
pip install pytest httpx

# For coverage reports
pip install pytest-cov

# For async testing
pip install pytest-asyncio
```

### Basic Test Execution

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run specific test method
pytest tests/test_api.py::TestRLIDSAPI::test_root_endpoint -v
```

### Coverage Analysis

```bash
# Run tests with coverage
pytest tests/ --cov=rl_ids --cov=api

# Generate HTML coverage report
pytest tests/ --cov=rl_ids --cov=api --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Test Output Examples

**Successful Test Run**
```
tests/test_api.py::TestRLIDSAPI::test_root_endpoint PASSED
tests/test_api.py::TestRLIDSAPI::test_health_check PASSED
tests/test_api.py::TestRLIDSAPI::test_model_info PASSED
tests/test_api.py::TestRLIDSAPI::test_single_prediction PASSED

================= 4 passed in 2.34s =================
```

**Coverage Report**
```
Name                     Stmts   Miss  Cover
--------------------------------------------
api/__init__.py              0      0   100%
api/main.py                 45      5    89%
api/models.py               23      2    91%
api/services.py             34      8    76%
--------------------------------------------
TOTAL                      102     15    85%
```

## Test Configuration

### Environment Setup

Create a test-specific environment configuration:

```bash
# Create .env.test file
cat > .env.test << EOF
# Test Environment Configuration
API_HOST=localhost
API_PORT=8001
DEBUG=true
LOG_LEVEL=DEBUG

# Use test models/data
MODEL_PATH=tests/fixtures/test_model.pt
TEST_MODE=true
EOF
```

### Test Fixtures

Example test fixtures for consistent testing:

```python
# conftest.py (if exists)
import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
def sample_features():
    """Sample feature vector for testing."""
    return [0.1] * 78  # 78 features as expected by model

@pytest.fixture
def mock_model_response():
    """Mock model prediction response."""
    return {
        "prediction": "BENIGN",
        "confidence": 0.95,
        "attack_type": None
    }
```

## Writing Tests

### Test Structure Guidelines

```python
def test_function_name():
    """
    Test description explaining what is being tested.
    
    This test verifies that [specific functionality] works correctly
    when [specific conditions] are met.
    """
    # Arrange - Set up test data and conditions
    
    # Act - Execute the functionality being tested
    
    # Assert - Verify the results
```

### API Test Example

```python
def test_health_check(self):
    """Test the health check endpoint returns correct status."""
    # Arrange
    expected_keys = ["status", "timestamp", "details"]
    
    # Act
    response = self.client.get("/health")
    data = response.json()
    
    # Assert
    assert response.status_code == 200
    assert data["status"] == "healthy"
    assert all(key in data for key in expected_keys)
    assert data["details"]["model_loaded"] is True
```

### Model Test Example

```python
def test_model_prediction_format():
    """Test that model predictions return expected format."""
    # Arrange
    from rl_ids.agents.dqn_agent import DQNAgent
    agent = DQNAgent()
    sample_input = torch.randn(1, 78)
    
    # Act
    prediction = agent.predict(sample_input)
    
    # Assert
    assert isinstance(prediction, dict)
    assert "class" in prediction
    assert "confidence" in prediction
    assert 0 <= prediction["confidence"] <= 1
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```python
# Test feature extraction
def test_feature_extraction():
    from rl_ids.make_dataset import extract_features
    # Test individual feature extraction functions

# Test agent components
def test_dqn_network():
    from rl_ids.agents.dqn_agent import DQNNetwork
    # Test neural network components

# Test utility functions
def test_config_loading():
    from rl_ids.config import load_config
    # Test configuration management
```

### Integration Tests

Test component interactions:

```python
# Test full prediction pipeline
def test_end_to_end_prediction():
    # Test: Raw data → Features → Model → Prediction

# Test API with real model
def test_api_model_integration():
    # Test API endpoints with actual model loading

# Test monitoring pipeline
def test_network_monitoring_flow():
    # Test: Packet capture → Feature extraction → Detection
```

### Performance Tests

Test system performance characteristics:

```python
import time
import pytest

def test_prediction_latency():
    """Test that predictions complete within acceptable time."""
    start_time = time.time()
    
    # Make prediction
    response = client.post("/predict", json=sample_data)
    
    end_time = time.time()
    latency = end_time - start_time
    
    assert response.status_code == 200
    assert latency < 0.1  # Less than 100ms

@pytest.mark.performance
def test_batch_prediction_throughput():
    """Test batch prediction throughput."""
    batch_size = 100
    batch_data = [sample_features] * batch_size
    
    start_time = time.time()
    response = client.post("/predict/batch", json=batch_data)
    end_time = time.time()
    
    throughput = batch_size / (end_time - start_time)
    assert throughput > 50  # At least 50 predictions per second
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12, 3.13]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=rl_ids --cov=api
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Test Data Management

### Mock Data Generation

```python
def generate_test_features(attack_type="benign", count=100):
    """Generate realistic test features."""
    if attack_type == "benign":
        # Generate normal traffic patterns
        features = np.random.normal(0, 1, (count, 78))
    elif attack_type == "ddos":
        # Generate DDoS-like patterns
        features = np.random.normal(2, 0.5, (count, 78))
    
    return features.tolist()
```

### Test Dataset Management

```python
# tests/fixtures/data.py
import pandas as pd

def load_test_dataset():
    """Load small test dataset for consistent testing."""
    return pd.read_csv("tests/fixtures/test_sample.csv")

def create_test_environment_data():
    """Create test data for RL environment."""
    states = generate_test_features("benign", 50)
    actions = [0] * 25 + [1] * 25  # Mix of normal and attack classifications
    rewards = [1.0] * 50  # All correct classifications
    
    return states, actions, rewards
```

## Debugging Tests

### Common Issues and Solutions

#### Test Failures
```bash
# Run failed tests only
pytest --lf

# Run tests with detailed output
pytest -vv --tb=long

# Drop into debugger on failure
pytest --pdb
```

#### Model Loading Issues
```python
# Skip tests if model not available
@pytest.mark.skipif(not os.path.exists("models/dqn_model_best.pt"), 
                   reason="Model file not found")
def test_model_prediction():
    # Test code here
```

#### Async Test Issues
```python
# For testing async endpoints
@pytest.mark.asyncio
async def test_async_endpoint():
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/async-endpoint")
    assert response.status_code == 200
```

## Test Metrics and Quality

### Coverage Goals
- **Minimum Coverage**: 80% overall
- **Critical Components**: 95% coverage
- **API Endpoints**: 100% coverage

### Quality Metrics
- **Test Execution Time**: < 30 seconds for full suite
- **Flaky Test Rate**: < 5%
- **Test Maintenance**: Regular updates with code changes

## Best Practices

### Test Writing
1. **Clear Test Names**: Describe what is being tested
2. **Single Responsibility**: One assertion per test concept
3. **Arrange-Act-Assert**: Clear test structure
4. **Independent Tests**: No dependencies between tests

### Test Maintenance
1. **Regular Updates**: Keep tests updated with code changes
2. **Cleanup**: Remove obsolete tests
3. **Documentation**: Document complex test scenarios
4. **Performance**: Monitor test execution time

### Test Data
1. **Deterministic**: Use fixed seeds for reproducible results
2. **Realistic**: Use data that represents real-world scenarios
3. **Isolated**: Don't depend on external resources
4. **Clean**: Clean up test artifacts after execution

## Running Specific Test Scenarios

### API Testing
```bash
# Test only API endpoints
pytest tests/test_api.py -v

# Test specific API functionality
pytest tests/test_api.py -k "prediction" -v
```

### Model Testing
```bash
# Test model-related functionality
pytest tests/ -k "model" -v

# Test with different model configurations
MODEL_CONFIG=test pytest tests/ -v
```

### Performance Testing
```bash
# Run performance tests
pytest tests/ -m performance -v

# Run with performance profiling
pytest tests/ --profile -v
```