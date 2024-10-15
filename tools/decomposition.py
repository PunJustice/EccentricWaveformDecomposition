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


def PeriastronIndicesFrom20(h22, IDs, include_ends=False):
    indices = {}
    N = np.size(IDs)
    for k in range(N):
        temp = scipy.signal.find_peaks(-np.real(h22[IDs[k]]["h20"]))
        temp = list(temp)[0]
        if include_ends:
            temp = np.append([0], temp)
            temp = np.append(temp, [len(h22[IDs[k]]["t"]) - 1])
        indices[IDs[k]] = temp
    return indices


def PeriastronIndicesFromQC(h22, IDs, include_ends=False):
    indices = {}
    N = np.size(IDs)
    A22_QC = np.abs(h22["QC"]["h22"])
    t_QC = h22["QC"]["t"]
    A22_QC_interp = scipy.interpolate.interp1d(t_QC, A22_QC)
    for k in range(N):
        A22_QC_interpolated = A22_QC_interp(h22[IDs[k]]["t"])
        temp = scipy.signal.find_peaks(np.abs(h22[IDs[k]]["h22"]) - A22_QC_interpolated)
        temp = list(temp)[0]
        if h22[IDs[k]]["t"][temp[-1]] > -10.0:
            temp = temp[:-1]
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


def AddMeanAnomalyDomain(h22, IDs, periastron_definition="Amp"):
    N = np.size(IDs)
    if periastron_definition == "Amp":
        periastron_indices = PeriastronIndices(h22, IDs, include_ends=True)
    elif periastron_definition == "h20":
        periastron_indices = PeriastronIndicesFrom20(h22, IDs, include_ends=True)
    elif periastron_definition == "QC":
        periastron_indices = PeriastronIndicesFromQC(h22, IDs, include_ends=True)
    else:
        Exception("Definition of periastron not supported!")
    for k in range(N):
        periastron_times = h22[IDs[k]]["t"][periastron_indices[IDs[k]]]
        temp = ComputeMeanAnomalyArray(h22[IDs[k]]["t"], periastron_times)
        temp = np.unwrap(temp)
        h22[IDs[k]]["MeanAno"] = temp - temp[-1]
    return h22


def MeasureReferenceEccandMeanAno(h22, IDs, e_tref=-2000, l_tref=-1000):
    ecc_object = {}
    l_object = {}
    N = np.size(IDs)

    for k in range(N):
        h22_raw = h22[IDs[k]]["h22"]
        t = h22[IDs[k]]["t"]
        Reh22 = np.real(np.array(h22_raw))
        Imh22 = np.imag(np.array(h22_raw))
        re_interp = scipy.interpolate.InterpolatedUnivariateSpline(t, Reh22)
        im_interp = scipy.interpolate.InterpolatedUnivariateSpline(t, Imh22)
        t_uniform = np.arange(min(t), max(t), step=0.1)
        h22_uniform = re_interp(t_uniform) + 1.0j * im_interp(t_uniform)
        dataDict = {}
        dataDict["t"] = t_uniform
        dataDict["hlm"] = {}
        dataDict["hlm"][(2, 2)] = h22_uniform
        test = gw_eccentricity.measure_eccentricity(
            tref_in=[e_tref, l_tref],
            dataDict=dataDict,
            method="AmplitudeFits",
            num_orbits_to_exclude_before_merger=0,
        )
        ecc_object[IDs[k]] = test["eccentricity"][0]
        l_object[IDs[k]] = test["mean_anomaly"][1]

    return ecc_object, l_object


def InterpolateToUniformMeanAnomalyDomain(
    h22, IDs, dl=0.001, l_min=-30 * np.pi, l_max=0
):
    N = np.size(IDs)
    N_points = int(l_max - l_min / dl) + 1
    meanano_domain = np.linspace(l_min, l_max, N_points)
    for k in range(N):
        A22_interp = scipy.interpolate.InterpolatedUnivariateSpline(
            h22[IDs[k]]["MeanAno"], h22[IDs[k]]["A22"]
        )
        phi22_interp = scipy.interpolate.InterpolatedUnivariateSpline(
            h22[IDs[k]]["MeanAno"], h22[IDs[k]]["phi22"]
        )
        t_interp = scipy.interpolate.InterpolatedUnivariateSpline(
            h22[IDs[k]]["MeanAno"], h22[IDs[k]]["t"]
        )
        h22[IDs[k]]["A22_uniform"] = A22_interp(meanano_domain)
        h22[IDs[k]]["phi22_uniform"] = phi22_interp(meanano_domain)
        h22[IDs[k]]["t_uniform"] = t_interp(meanano_domain)
        h22[IDs[k]]["MeanAno_uniform"] = meanano_domain
    return h22
