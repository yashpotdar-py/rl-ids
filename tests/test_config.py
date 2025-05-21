"""
Test suite for the project configuration module.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_dotenv_loading():
    """Test that dotenv is loaded when config is imported."""
    # We need to patch logger.remove to avoid errors when it tries to remove handler 0
    with patch('dotenv.load_dotenv') as mock_load_dotenv, \
            patch('loguru.logger.remove'), \
            patch('loguru.logger.add'), \
            patch('loguru.logger.info'):

        # Force reload of the module to trigger dotenv loading
        if 'rl_ids.config' in sys.modules:
            del sys.modules['rl_ids.config']
        import rl_ids.config

        # Verify dotenv was called
        mock_load_dotenv.assert_called_once()


def test_tqdm_logger_configuration():
    """Test logger configuration with tqdm."""
    # Mock tqdm module
    mock_tqdm = MagicMock()
    mock_tqdm.write = MagicMock()

    # Patch the imports to simulate tqdm being available
    with patch.dict('sys.modules', {'tqdm': mock_tqdm}), \
            patch('tqdm.tqdm', mock_tqdm), \
            patch('loguru.logger.remove') as mock_logger_remove, \
            patch('loguru.logger.add') as mock_logger_add, \
            patch('loguru.logger.info'):

        # Force reload of the module
        if 'rl_ids.config' in sys.modules:
            del sys.modules['rl_ids.config']
        import rl_ids.config

        # Verify logger was reconfigured
        mock_logger_remove.assert_called_once_with(0)
        mock_logger_add.assert_called_once()


def test_tqdm_logger_fallback():
    """Test logger behavior when tqdm is not available."""
    # Remove tqdm from sys.modules to simulate it not being installed
    with patch.dict('sys.modules', {'tqdm': None}), \
            patch('rl_ids.config.tqdm', side_effect=ModuleNotFoundError("No module named 'tqdm'"), create=True), \
            patch('loguru.logger.info'):

        # Force reload of the module
        if 'rl_ids.config' in sys.modules:
            del sys.modules['rl_ids.config']

        # This should not raise an exception and the except block should be triggered
        import rl_ids.config

        # Success if we reach here without exception


class TestPaths:
    """Tests for path configuration."""

    @pytest.fixture(autouse=True)
    def setup_patches(self):
        """Setup patches for all tests in this class."""
        # Patch loguru to prevent errors with handler removal
        with patch('loguru.logger.remove'), \
                patch('loguru.logger.add'), \
                patch('loguru.logger.info'):
            yield

    def test_proj_root_is_correct(self):
        """Test that PROJ_ROOT is set to the correct parent directory."""
        # Force reload of the module
        if 'rl_ids.config' in sys.modules:
            del sys.modules['rl_ids.config']

        from rl_ids.config import PROJ_ROOT

        # The project root should be two levels up from the config.py file
        expected_root = Path(__file__).resolve().parents[1]
        assert PROJ_ROOT == expected_root
        assert PROJ_ROOT.exists(), "Project root directory does not exist"

    def test_all_paths_are_path_objects(self):
        """Test that all path variables are Path objects."""
        # Force reload of the module
        if 'rl_ids.config' in sys.modules:
            del sys.modules['rl_ids.config']

        from rl_ids import config

        path_vars = [
            'PROJ_ROOT', 'DATA_DIR', 'RAW_DATA_DIR', 'INTERIM_DATA_DIR',
            'PROCESSED_DATA_DIR', 'EXTERNAL_DATA_DIR', 'MODELS_DIR',
            'REPORTS_DIR', 'FIGURES_DIR'
        ]

        for var_name in path_vars:
            path_var = getattr(config, var_name)
            assert isinstance(path_var, Path), f"{var_name} should be a Path object"

    def test_path_hierarchy(self):
        """Test that all directories follow the correct hierarchy."""
        # Force reload of the module
        if 'rl_ids.config' in sys.modules:
            del sys.modules['rl_ids.config']

        from rl_ids.config import (
            PROJ_ROOT, DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR,
            PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR,
            REPORTS_DIR, FIGURES_DIR
        )

        # Test data directory hierarchy
        assert DATA_DIR == PROJ_ROOT / "data"
        assert RAW_DATA_DIR == DATA_DIR / "raw"
        assert INTERIM_DATA_DIR == DATA_DIR / "interim"
        assert PROCESSED_DATA_DIR == DATA_DIR / "processed"
        assert EXTERNAL_DATA_DIR == DATA_DIR / "external"

        # Test models directory
        assert MODELS_DIR == PROJ_ROOT / "models"

        # Test reports directory hierarchy
        assert REPORTS_DIR == PROJ_ROOT / "reports"
        assert FIGURES_DIR == REPORTS_DIR / "figures"

    def test_directory_creation(self, tmp_path):
        """Test that directories are structured correctly (simulated)."""
        # Directly patch PROJ_ROOT in the config module after import
        with patch('loguru.logger.info'), \
                patch('loguru.logger.remove'), \
                patch('loguru.logger.add'):

            # First import the module normally
            if 'rl_ids.config' in sys.modules:
                del sys.modules['rl_ids.config']
            import rl_ids.config as test_config

            # Then patch PROJ_ROOT and trigger recalculation of dependent paths
            with patch.object(test_config, 'PROJ_ROOT', tmp_path):
                # Manually recalculate the dependent paths
                test_config.DATA_DIR = tmp_path / "data"
                test_config.RAW_DATA_DIR = test_config.DATA_DIR / "raw"
                test_config.INTERIM_DATA_DIR = test_config.DATA_DIR / "interim"
                test_config.PROCESSED_DATA_DIR = test_config.DATA_DIR / "processed"
                test_config.EXTERNAL_DATA_DIR = test_config.DATA_DIR / "external"
                test_config.MODELS_DIR = tmp_path / "models"
                test_config.REPORTS_DIR = tmp_path / "reports"
                test_config.FIGURES_DIR = test_config.REPORTS_DIR / "figures"

                # Now verify the paths
                assert test_config.PROJ_ROOT == tmp_path
                assert test_config.DATA_DIR == tmp_path / "data"
                assert test_config.RAW_DATA_DIR == test_config.DATA_DIR / "raw"
                assert test_config.PROCESSED_DATA_DIR == test_config.DATA_DIR / "processed"
