"""Smoke tests so pytest runs before algorithms are implemented."""


def test_package_imports():
    import mlpackage

    assert mlpackage.__version__
