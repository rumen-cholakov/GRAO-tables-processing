from json import load
from regex import search
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from grao_tables_processing.common.regex_pattern_wrapper import RegexPatternWrapper
from grao_tables_processing.common.custom_types import DataTuple, HeaderEnum, TableTypeEnum


@dataclass
class Configuration():
  data_configuration_path: str
  processed_tables_path: str
  matched_tables_path: str
  combined_tables_path: str
  visualizations_path: str
  pickled_data_path: str
  credentials_path: str
  data: List[str] = field(init=False)
  _extra_params: Dict[Any, Any] = field(default_factory=dict)

  def __post_init__(self):
    with open(self.data_configuration_path) as file:
      self.data = load(file)

  def __getitem__(self, key: Any):
    return self._extra_params.get(key, None)

  def __setitem__(self, key: Any, value: Any):
    self._extra_params[key] = value

  def process_data_configuration(self) -> List[DataTuple]:
    output = []
    for entry in self.data:
      output.append(Configuration._data_tuple_from_entry(entry))

    return output

  @staticmethod
  def _data_tuple_from_entry(entry: str) -> Optional[DataTuple]:
    if search(RegexPatternWrapper().date_group, entry):
      return DataTuple(entry, HeaderEnum.New, TableTypeEnum.Quarterly)

    if date := search(RegexPatternWrapper().year_group, entry):
      header_type = HeaderEnum(int(date.group(1)) > 2005)
      return DataTuple(entry, header_type, TableTypeEnum.Yearly)
