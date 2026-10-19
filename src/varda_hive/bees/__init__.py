"""The bees. Importing this package registers all their checks."""
from .meta import checks as _meta  # noqa: F401
from .spot import checks as _spot  # noqa: F401
from .stats import checks as _stats  # noqa: F401
