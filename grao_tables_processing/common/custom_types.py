from enum import IntEnum
from datetime import datetime as dt_class
from typing import Any, Dict, Tuple, TypeVar, NamedTuple


class HeaderEnum(IntEnum):
  Old = 0
  New = 1


class TableTypeEnum(IntEnum):
  Quarterly = 0
  Yearly = 1


class DataTuple(NamedTuple):
  data: Any
  header_type: HeaderEnum
  table_type: TableTypeEnum


class SettlementDataTuple(NamedTuple):
  key: Tuple[str, str, str]
  data: Any


class SettlementNamesForPeriod(NamedTuple):
  name: Tuple[str, ...]
  start: dt_class
  end: dt_class


class MunicipalityIdentifier(NamedTuple):
  region: str
  municipality: str


class SettlementInfo(NamedTuple):
  name: str
  permanent_residents: int
  current_residents: int


class FullSettlementInfo(NamedTuple):
  region: str
  municipality: str
  settlement: str
  permanent_residents: int
  current_residents: int


class ParsedLines(NamedTuple):
  municipality_ids: Dict[int, MunicipalityIdentifier]
  settlements_info: Dict[int, SettlementInfo]


T = TypeVar('T')
U = TypeVar('U')
