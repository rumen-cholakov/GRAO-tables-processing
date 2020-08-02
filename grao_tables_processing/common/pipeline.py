from typing import Generic, Sequence, Callable
from functools import reduce

from grao_tables_processing.common.custom_types import T


class Pipeline(Generic[T]):
  def __init__(self, functions: Sequence[Callable[[T], T]]):
    self.functions_sequence = functions

  def __call__(self, value: T) -> T:
    return reduce(Pipeline._apply, self.functions_sequence, value)

  @staticmethod
  def _apply(val: T, func: Callable[[T], T]) -> T:
    return func(val)
