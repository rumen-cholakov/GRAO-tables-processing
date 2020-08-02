from typing import Sequence, Callable, Optional
from functools import reduce

from grao_tables_processing.common.custom_types import T, U


class Pipeline():
  def __init__(self, functions: Sequence[Callable[[T], Optional[T]]]):
    self.functions_sequence = functions

  def __call__(self, value: T) -> T:
    return reduce(Pipeline._apply, self.functions_sequence, value)

  @staticmethod
  def _apply(val: T, func: Callable[[T], U]) -> U:
    return func(val)
