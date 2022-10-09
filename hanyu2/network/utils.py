import enum
import numpy as np

class WindowType(enum.Enum):
  HAMMING = 1
  HANNING = 2
  SQUARED_HANNING = 3

def window(window_type: WindowType, tmax: int):
  """Window functions generator.

  Creates a window of type window_type and duration tmax.
  Currently, hanning (also known as Hann) and hamming windows are available.

  Args:
    window_type: str, type of window function (hanning, squared_hanning,
      hamming)
    tmax: int, duration of the window, in samples

  Returns:
    a window function as np array
  """

  def hanning(n: int):
    return 0.5 * (1 - np.cos(2 * np.pi * (n - 1) / (tmax - 1)))

  def hamming(n: int):
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / tmax)

  if window_type == WindowType.HANNING:
    return np.asarray([hanning(n) for n in range(tmax)])
  elif window_type == WindowType.SQUARED_HANNING:
    return np.asarray([hanning(n) for n in range(tmax)])**2
  elif window_type == WindowType.HAMMING:
    return np.asarray([hamming(n) for n in range(tmax)])
  else:
    raise ValueError('Wrong window type.')
