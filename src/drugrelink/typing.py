# -*- coding: utf-8 -*-

from typing import Callable, Iterable, List, Tuple

import gensim

EmbedderFunction = Callable[[gensim.models.Word2Vec, Iterable[Tuple[str, str]]], List[List[float]]]
