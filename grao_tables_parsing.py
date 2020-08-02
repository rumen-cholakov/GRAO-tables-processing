#! /usr/bin/env python3.8

"""## Imports"""
import argparse
import os

from typing import Callable, Generic, List, Optional
from dataclasses import dataclass

from grao_tables_processing.common.custom_types import T, U
from grao_tables_processing import Configuration
from grao_tables_processing import PickleWrapper

from grao_tables_processing import settlement_disambiguation
from grao_tables_processing import table_parser
from grao_tables_processing import create_table_processor
from grao_tables_processing import create_visualizations
from grao_tables_processing import update_matched_data, update_all_settlements


"""## Input validation """


@dataclass
class ValidationItem(Generic[T]):
  parameter: T
  action: Callable[[T], U]
  check: Callable[[T], bool]

  def execute_action(self):
    return self.action(self.parameter)

  def execute_check(self):
    return self.check(self.parameter)


def input_validation_callback(
  message: str,
  return_vale: T = None,
  action: Optional[Callable[[], Optional[T]]] = None
) -> Optional[T]:

  print(message)
  result = None

  if action:
    result = action()

  if return_vale is not None:
    result = return_vale

  return result


def make_dir(path: str) -> bool:
  result = input_validation_callback(
    f'Creating directory at path: {path}',
    return_vale=True,
    action=(lambda: os.makedirs(path))
  )

  return result or False


def signal_for_missing_file(path: str) -> bool:
  result = input_validation_callback(
    f'ERROR: File at {path} is missing!!!',
    return_vale=False
  )

  return result or False


def validate_input(input_list: List[ValidationItem]) -> bool:
  results = [validation_item.execute_action() for validation_item in input_list if not validation_item.execute_check()]

  return all(results)


def main():

  current_dir = os.path.dirname(os.path.abspath(__file__))

  example_text = """Examples:
    python3  grao_tables_processing.py

    python3  grao_tables_processing.py
      --data_configuration_path <path to file>
      --processed_tables_path <path to folder>
      --matched_tables_path <path to folder>
      --combined_tables_path <path to folder>
      --visualizations_path <path to folder>
      --pickled_data_path <path to folder>
      --credentials_path <path to file>
      --produce_graphics
      --update_wiki_data
  """

  parser = argparse.ArgumentParser(description="Processes the tables provided by GRAO and"
                                               "extracts the information from them to csv files",
                                   epilog=example_text,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--data_configuration_path",
                      type=str, default=f'{current_dir}/config/data_config.json',
                      help="Path to the JSON file containing the configuration for which tables should be processed.")
  parser.add_argument("--processed_tables_path",
                      type=str, default=f'{current_dir}/grao_data',
                      help="Path to the folder where the processed tables will be stored.")
  parser.add_argument("--matched_tables_path",
                      type=str, default=f'{current_dir}/matched_data',
                      help="Path to the folder where the matched tables are be stored.")
  parser.add_argument("--combined_tables_path",
                      type=str, default=f'{current_dir}/combined_tables',
                      help="Path to the folder where the combined tables will be stored.")
  parser.add_argument("--visualizations_path",
                      type=str, default=f'{current_dir}/visualizations',
                      help="Path to the folder where the combined tables will be stored.")
  parser.add_argument("--pickled_data_path",
                      type=str, default=f'{current_dir}/pickled_data',
                      help="Path to the folder where pickled objects will be stored.")
  parser.add_argument("--credentials_path",
                      type=str, default=f'{current_dir}/credentials/wd_credentials.csv',
                      help="Path to the file containing credentials.")
  parser.add_argument("--produce_graphics",
                      default=False, action="store_true",
                      help="If set the script will produce graphics from the processed tables.")
  parser.add_argument("--update_wiki_data",
                      default=False, action="store_true",
                      help="If set the script will update WikiData with the processed tables.")

  args = parser.parse_args()

  validation_result = validate_input([
    ValidationItem(args.data_configuration_path,
                   signal_for_missing_file,
                   os.path.exists),
    ValidationItem(args.processed_tables_path,
                   make_dir,
                   os.path.exists),
    ValidationItem(args.matched_tables_path,
                   make_dir,
                   os.path.exists),
    ValidationItem(args.combined_tables_path,
                   make_dir,
                   os.path.exists),
    ValidationItem(args.visualizations_path,
                   make_dir,
                   os.path.exists),
    ValidationItem(args.pickled_data_path,
                   make_dir,
                   os.path.exists),
    ValidationItem(args.credentials_path,
                   signal_for_missing_file,
                   os.path.exists)
  ])

  if not validation_result:
    exit(1)

  PickleWrapper.configure(args.pickled_data_path)

  configuration = Configuration(
    args.data_configuration_path,
    args.processed_tables_path,
    args.matched_tables_path,
    args.combined_tables_path,
    args.visualizations_path,
    args.pickled_data_path,
    args.credentials_path
  )

  configuration['settlement_disambiguation'] = settlement_disambiguation
  configuration['table_parser'] = table_parser

  processing_pipeline = create_table_processor(configuration)
  data_source = configuration.process_data_configuration()

  processing_pipeline(data_source)

  if args.produce_graphics:
    create_visualizations(configuration)

  if args.update_wiki_data:
    update_matched_data(configuration)
    update_all_settlements(configuration)


if __name__ == "__main__":
  main()
