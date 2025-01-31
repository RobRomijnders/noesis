"""General utility functions for NoEsis."""
import subprocess
from typing import Iterable


def iter_datasets_from_str(x: str) -> Iterable[str]:
    """Iterates over datasets from a string."""
    yield from sorted(map(lambda x: x.lower().strip(), x.split(',')))

def has_multiple_datasets(x: str) -> bool:
    """Checks if a string has multiple datasets."""
    return ',' in x

def strip_datasets(x: str) -> str:
    """Strips datasets."""
    return '-'.join(iter_datasets_from_str(x))

def make_git_log():
    """Logs the git diff and git show.

    Note that this function has a general try/except clause and will except most
    errors produced by the git commands.
    """
    try:
        result = subprocess.run(
          ['git', 'status'], stdout=subprocess.PIPE, check=True)
        print(f"Git status \n{result.stdout.decode('utf-8')}")

        result = subprocess.run(
          ['git', 'show', '--summary'], stdout=subprocess.PIPE, check=True)
        print(f"Git show \n{result.stdout.decode('utf-8')}")

        result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE, check=True)
        print(f"Git diff \n{result.stdout.decode('utf-8')}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Git log not printed due to {e}")
