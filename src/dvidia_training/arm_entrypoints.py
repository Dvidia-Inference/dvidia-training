"""Friendly preflight for the optional native simulation commands."""
import importlib.util


def require_arm():
    if importlib.util.find_spec('mujoco') is None:
        raise ValueError('This movement/simulation tool requires the optional arm extra. From the repository, install it with: python -m pip install ".[arm]"')


def capsule_main():
    try:
        require_arm()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    from .capsule_installer import main
    main()
