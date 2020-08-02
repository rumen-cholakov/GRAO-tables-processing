from typing import Any, Optional
from pickle import dump, load
from os.path import exists
from os import makedirs


class PickleWrapper():

  directory = ''

  @staticmethod
  def configure(directory: str):
    PickleWrapper.directory = directory

  @staticmethod
  def pickle_data(data: Any, name: str):
    directory = PickleWrapper.directory

    if not exists(directory):
      makedirs(directory)

    with open(f'{directory}/{name}.pkl', 'wb') as f:
      dump(data, f)

  @staticmethod
  def load_data(name: str) -> Optional[Any]:
    directory = PickleWrapper.directory
    path = f'{directory}/{name}.pkl'

    if not exists(path):
      return None

    with open(path, 'rb') as f:
      return load(f)
