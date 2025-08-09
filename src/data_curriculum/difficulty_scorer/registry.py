from typing import Mapping, Type, TypeVar

from .base_difficulty_scorer import BaseDifficultyScorer

# Define a type variable T, constrained to subclasses of BaseDifficultyScorer.
# This allows us to write generic code that works with any subclass of BaseDifficultyScorer.
T = TypeVar("T", bound=BaseDifficultyScorer)

# Global registry mapping string names to difficulty scorer classes
DIFFICULTY_SCORER_REGISTRY: Mapping[str, Type[BaseDifficultyScorer]] = {}


def register_difficulty_scorer(name: str):
    """
    Decorator function to register a new difficulty scorer class into the global registry.

    Args:
        name (str): A unique string name for the scorer, used as a key in DIFFICULTY_SCORER_REGISTRY.

    Returns:
        Callable: A decorator that, when applied to a class, will add it to the registry.
    """
    def _register(cls: Type[T]) -> Type[T]:
        """
        Inner function that actually registers the class in the registry.

        Args:
            cls (Type[T]): The class to register, must be a subclass of BaseDifficultyScorer.

        Returns:
            Type[T]: The same class, unchanged.
        """
        DIFFICULTY_SCORER_REGISTRY[name] = cls
        return cls

    return _register