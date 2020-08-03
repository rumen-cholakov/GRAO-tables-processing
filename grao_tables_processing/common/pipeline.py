from typing import Generic, Optional, Iterable, Callable, Any
from functools import reduce

from grao_tables_processing.common.custom_types import T, UnexpectedNoneError

PipelineFunctionType = Callable[[T], T]
PipelineFunctionDecorator = Callable[[PipelineFunctionType[T]], PipelineFunctionType[T]]


class Pipeline(Generic[T]):
  def __init__(self, functions: Iterable[Callable[[T], T]]):
    self.base_functions = functions
    self.functions_iterable = functions
    self.logger: Optional[Any] = None

  def __call__(self, value: T) -> T:
    return reduce(Pipeline._apply, self.functions_iterable, value)

  def activate_verbose_logging(self, logger: Any):
    self.logger = logger
    self.functions_iterable = map(self._add_logging_decorator, self.functions_iterable)

  def deactivate_verbose_logging(self):
    self.functions_iterable = self.base_functions

  def add_decorator(self, decorator: PipelineFunctionDecorator):
    self.functions_iterable = map(decorator, self.functions_iterable)

  @staticmethod
  def _apply(val: T, func: PipelineFunctionType) -> T:
    return func(val)

  def _add_logging_decorator(self, function: PipelineFunctionType) -> PipelineFunctionType:
    def wrapper_decorator(arg: T) -> T:
      if self.logger is None:
        raise UnexpectedNoneError('Tried to log from pipeline without a configured Logger!')

      self.logger.print('function: ', function.__name__)
      self.logger.print('arguments: ', arg)
      result = function(arg)
      self.logger.print('result: ', result)
      return result
    return wrapper_decorator
