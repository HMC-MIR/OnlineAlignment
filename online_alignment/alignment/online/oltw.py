"""Online Time Warping algorithm per Dixon et al."""

from ..base import OnlineAlignment


class OLTW(OnlineAlignment):
    """Online Time Warping Algorithm."""

    def __init__(self, *args, **kwargs):
        """Initialize OLTW algorithm.

        Args:
            *args: Arguments passed to OnlineAlignment base class.
            **kwargs: Keyword arguments passed to OnlineAlignment base class.
        """
        super().__init__(*args, **kwargs)
        raise NotImplementedError
