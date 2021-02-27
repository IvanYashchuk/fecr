import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--fenics", action="store_true", default=False, help="run fenics backend tests"
    )
    parser.addoption(
        "--firedrake",
        action="store_true",
        default=False,
        help="run firedrake backend tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "fenics: mark test as a fenics test")
    config.addinivalue_line("markers", "firedrake: mark test as a firedrake test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--fenics") and config.getoption("--firedrake"):
        # --fenics and --firedrake given in cli: do not skip tests
        return
    skip_fenics = pytest.mark.skip(reason="need --fenics option to run")
    skip_firedrake = pytest.mark.skip(reason="need --firedrake option to run")
    for item in items:
        if not config.getoption("--fenics"):
            if "fenics" in item.keywords:
                item.add_marker(skip_fenics)
        if not config.getoption("--firedrake"):
            if "firedrake" in item.keywords:
                item.add_marker(skip_firedrake)
