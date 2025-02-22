# 🧪 Testing

## Running Tests

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=cli --cov-report=term-missing

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
```

### Test Structure

1. **Unit Tests**
   - Game interface functionality
   - Leaderboard system
   - Configuration management
   - Reward calculations

2. **Integration Tests**
   - Contract interaction
   - Stake management
   - Score submission
   - Error handling

### Writing Tests

1. **Unit Tests**: Add new unit tests in `tests/unit/`

   ```python
   def test_your_feature():
       # Arrange
       game = YourGame()
       
       # Act
       result = game.your_method()
       
       # Assert
       assert result == expected_value
   ```

2. **Integration Tests**: Add new integration tests in `tests/integration/`

   ```python
   def test_your_integration(mock_wallet):
       # Mock external dependencies
       with patch('subprocess.run') as mock_run:
           # Set up mock response
           mock_process = MagicMock()
           mock_process.returncode = 0
           mock_process.stdout = "expected output"
           mock_run.return_value = mock_process
           
           # Execute and verify
           result = your_integration_method()
           assert result == expected_value
   ```

3. **Test Data**: Add test data files in `tests/data/`

### Test Coverage

We maintain high test coverage to ensure code quality:

- Core functionality: 90%+ coverage
- Game implementations: 80%+ coverage
- Contract integration: 85%+ coverage

Run coverage report to check current status:

```bash
pytest --cov=cli --cov-report=html
# View detailed report in htmlcov/index.html
```
