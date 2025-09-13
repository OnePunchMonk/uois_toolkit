import pytest
import os
import logging

logger = logging.getLogger(__name__)

def pytest_addoption(parser):
    """Adds the --dataset_path option to the pytest command line."""
    parser.addoption(
        "--dataset_path",
        action="append",
        default=[],
        help="Specify dataset paths in the format name=path (e.g., tabletop=/data/tabletop)",
    )

@pytest.fixture(scope="session")
def dataset_paths(request):
    """Parses the --dataset_path arguments and returns a dictionary."""
    path_dict = {}
    for arg in request.config.getoption("--dataset_path"):
        try:
            name, path = arg.split("=", 1)
            if os.path.exists(path):
                path_dict[name] = path
            else:
                logger.warning(f"Path for dataset '{name}' does not exist: {path}. It will be skipped.")
        except ValueError:
            pytest.fail(f"Invalid format for --dataset_path: '{arg}'. Use 'name=path'.")
    return path_dict
