from typing import Tuple, Union

from .orderedset import OrderedSet

Element = Tuple[Union[int, float, str, None], ...]
IndexingSet = OrderedSet[Element]
