import matplotlib.pyplot as plt  # type: ignore

from typing import Any, Dict, Iterable, List, Tuple
from os.path import exists
from os import makedirs
from numpy import arange  # type: ignore

from grao_tables_processing.common.pickle_wrapper import PickleWrapper
from grao_tables_processing.common.configuration import Configuration
from grao_tables_processing.common.helper_functions import force_unwrap_optional


def autolabel(ax, rects: Iterable[Any]):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def load_processed_data() -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
  ekatte_to_triple = PickleWrapper.load_data('ekatte_to_triple')
  combined = PickleWrapper.load_data('combined_tables')

  ekatte_to_triple_unwrapped: Any = force_unwrap_optional(ekatte_to_triple, 'There was an issue loading the data!')
  combined_unwrapped: Any = force_unwrap_optional(combined, 'There was an issue loading the data!')

  combined_dict = combined_unwrapped.to_dict(orient='index')

  return ekatte_to_triple_unwrapped, combined_dict


def path_for_settlement_graphic(directory: str, name: str, suffix: str = '') -> str:
  modified_name = name.replace('.', '').replace(' ', '_')

  if suffix == '_':
    suffix = ''

  return f'{directory}/{modified_name}{suffix}'


def prepare_directory(triple: Tuple[str, str, str], base: str) -> str:
  sub_path = f'{triple[0]}/{triple[1]}'.replace(' ', '_')
  full_path = f'{base}/{sub_path}'

  if not exists(full_path):
    makedirs(full_path)

  return full_path


def clear_figure():
  # Clear the current axes.
  plt.cla()

  # Clear the current figure.
  plt.clf()
  plt.ioff()


def draw_plot(directory: str, plot_name: str, type_name: str):
  plt_path = path_for_settlement_graphic(directory, plot_name.split(',')[-1].strip(), f'_{type_name.lower()}')
  plt.savefig(plt_path)
  print(plt_path)
  clear_figure()


def plot_single_value(directory: str, plot_name: str, values: List[int], labels: List[str], type_name: str = ''):
  _, ax = plt.subplots()

  xticks = arange(len(labels))
  width = 0.4

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Number of residents')
  ax.set_xticks(xticks)
  ax.set_xticklabels(labels)
  ax.set_title(plot_name)

  rects = ax.bar(xticks - width / 20, values, width, label=type_name.capitalize(), align='center')
  autolabel(ax, rects)

  ax.legend()

  draw_plot(directory, plot_name, type_name)


def plot_comparison(
  directory: str,
  settlement_name: str,
  values_list: List[List[int]],
  labels: List[str],
  type_name: str = ''
):
  _, ax = plt.subplots()
  xticks = arange(len(labels))

  ax.set_xticks(xticks)
  ax.set_xticklabels(labels)

  for values in values_list:
    plt.plot(values)

  plt.title('Comparison between permanent and current')
  plt.xlabel('Year')
  plt.ylabel('Number of residents')
  plt.legend(['Permanent', 'Current'], loc='upper right')

  draw_plot(directory, settlement_name, type_name)


def create_visualizations(config: Configuration):
  ekatte_to_triple, combined_dict = load_processed_data()

  plt.rcParams['figure.figsize'] = [45, 15]

  labels = list(combined_dict[list(combined_dict.keys())[0]].keys())
  date_labels = list(map(lambda l: ' '.join(l.split('_')[1:]), labels[0::2]))
  date_labels.reverse()

  for item in combined_dict:
    triple = ekatte_to_triple[item]
    name = f'обл. {triple[0]}, общ. {triple[1]}, {triple[2]}'
    full_path = prepare_directory(triple, config.visualizations_path)

    values = list(combined_dict[item].values())

    permanent_values = values[0::2]
    permanent_values.reverse()
    plot_single_value(full_path, name, permanent_values, date_labels, 'permanent')

    current_values = values[1::2]
    current_values.reverse()
    plot_single_value(full_path, name, current_values, date_labels, 'current')

    plot_comparison(full_path, name, [permanent_values, current_values], date_labels, 'compare')

    # Closes all the figure windows.
    plt.close('all')
