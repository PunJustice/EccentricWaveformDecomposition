import numpy as np
import sxs
import scipy
import gw_eccentricity


def PeriastronIndices(h22, IDs, include_ends=False):
    indices = {}
    N = np.size(IDs)
    for k in range(N):
        temp = scipy.signal.find_peaks(np.abs(h22[IDs[k]]["h22"]))
        temp = list(temp)[0]
        if include_ends:
            temp = np.append([0], temp)
            temp = np.append(temp, [len(h22[IDs[k]]["t"]) - 1])
        indices[IDs[k]] = temp
    return indices


def ComputeMeanAnomaly(time, periastron_times):
    subtraction = periastron_times - time
    next_periastron = min(n for n in subtraction if n >= 0)
    index = np.argmin(np.abs(subtraction - next_periastron))
    return (
        2
        * np.pi
        * (periastron_times[index - 1] - time)
        / (periastron_times[index - 1] - periastron_times[index])
    )


def ComputeMeanAnomalyArray(times, periastron_times):
    N = np.size(times)
    result = np.zeros(N)
    for i in range(N):
        result[i] = ComputeMeanAnomaly(times[i], periastron_times)
    return result


def AddMeanAnomalyDomain(h22, IDs):
    N = np.size(IDs)
    periastron_indices = PeriastronIndices(h22, IDs, include_ends=True)
    for k in range(N):
        periastron_times = h22[IDs[k]]["t"][periastron_indices[IDs[k]]]
        temp = ComputeMeanAnomalyArray(h22[IDs[k]]["t"], periastron_times)
        temp = np.unwrap(temp)
        h22[IDs[k]]["MeanAno"] = temp - temp[-1]
    return h22
