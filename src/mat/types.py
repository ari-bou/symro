from typing import Tuple, Union

from symro.src.mat.orderedset import OrderedSet

Element = Tuple[Union[int, float, str, None], ...]
IndexingSet = OrderedSet[Element]
