"""Environment variable management for RAGWatch."""

import os


def load_env() -> None:
    """Load environment variables from .env if python-dotenv is installed.

    Silently does nothing if python-dotenv is not installed or .env doesn't exist.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def get_env(
    name: str, default: str | None = None, required: bool = False
) -> str | None:
    """Read an environment variable with optional required validation.

    Args:
        name: Environment variable name.
        default: Default value if not set.
        required: If True and the variable is not set, raise an error.

    Returns:
        The environment variable value, or default.

    Raises:
        EnvironmentError: If required=True and the variable is not set.
    """
    value = os.environ.get(name, default)
    if required and not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set.\n"
            f"For local development:\n"
            f"  1. cp .env.template .env\n"
            f"  2. docker compose up -d postgres\n"
            f"  3. Set {name} in your .env file or export it."
        )
    return value
