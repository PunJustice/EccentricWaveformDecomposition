import numpy as np
import sxs
import scipy
import gw_eccentricity


def LoadSingleData(path, N):
    wf = sxs.load(path + "/Strain_N" + str(N))
    meta = sxs.Metadata.from_file(path + "/metadata.txt")
    horizons = sxs.load(path + "/Horizons")

    return wf, meta, horizons


def LoadData(IDs, N, project, Lev="Lev3"):
    wf = {}
    meta = {}
    horizons = {}

    for k in IDs:
        wf[k], meta[k], horizons[k] = LoadSingleData(
            "data/" + project + "/" + k + "/" + Lev, N
        )

    return wf, meta, horizons


def RestrictTo22Mode(wf, IDs):
    h22 = {}
    N = np.size(IDs)
    for k in range(N):
        h22[IDs[k]] = {}
        h22[IDs[k]]["h22"] = np.array(wf[IDs[k]][:, 4])
        h22[IDs[k]]["t"] = np.array(wf[IDs[k]].t)
    return h22


def CutJunk(h22, IDs, junk_time=1000):
    N = np.size(IDs)
    for k in range(N):
        junk_index = np.argmin(np.abs(h22[IDs[k]]["t"] - junk_time))
        h22[IDs[k]]["h22"] = h22[IDs[k]]["h22"][junk_index:]
        h22[IDs[k]]["t"] = h22[IDs[k]]["t"][junk_index:]
    return h22


def RestrictToFirstPeriastron(h22, IDs):
    N = np.size(IDs)
    for k in range(N):
        peri = scipy.signal.find_peaks(np.abs(h22[IDs[k]]["h22"]))[0][0]
        h22[IDs[k]]["h22"] = h22[IDs[k]]["h22"][peri:]
        h22[IDs[k]]["t"] = h22[IDs[k]]["t"][peri:]
    return h22


def RestrictToInspiral(h22, IDs):
    N = np.size(IDs)
    for k in range(N):
        peak_index = np.argmax(np.abs(h22[IDs[k]]["h22"]))
        h22[IDs[k]]["h22"] = h22[IDs[k]]["h22"][:peak_index]
        h22[IDs[k]]["t"] = h22[IDs[k]]["t"][:peak_index]
    return h22


def Align(h22, IDs):
    N = np.size(IDs)
    for k in range(N):
        h22[IDs[k]]["t"] = h22[IDs[k]]["t"] - h22[IDs[k]]["t"][-1]
    return h22
