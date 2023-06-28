#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modified on June 2023

@author: Vardan Adibekyan
"""

import numpy as np
import pandas as pd
import math
import requests
from io import StringIO
import astropy.units as u
from astroquery.gaia import Gaia
import sys
from sympy import *

def extinction_with_stilism(l, b, distance):
    """
    This function is based on the "stilism_distance.py" code
    It determines the reddening based on the l, b galactic coordinates and the distance
    https://stilism.obspm.fr/scripts

    stilism_dist.py: Get reddening from stilism web site for stars with l, b and distance
    __author__ = "Nicolas Leclerc"
    __license__ = "GPL"
    __version__ = "1.0.0"
    __maintainer__ = "Nicolas Leclerc"
    __status__ = "Production"
    """

    # Stilism URL for reddening query
    url = "http://stilism.obspm.fr/reddening?frame=galactic&vlong={}&ulong=deg&vlat={}&ulat=deg&distance={}"

    # Send HTTP GET request to Stilism website with coordinates and distance
    res = requests.get(url.format(l, b, distance), allow_redirects=True)

    if res.ok:
        # Read the response content into a StringIO object
        file = StringIO(res.content.decode("utf-8"))

        # Read the data from StringIO into a pandas DataFrame
        dfstilism = pd.read_csv(file)

        # Extract the necessary parameters from the DataFrame
        distance_stilism = dfstilism["distance[pc]"][0]
        reddening_stilism = dfstilism["reddening[mag]"][0]
        distance_uncertainty_stilism = dfstilism["distance_uncertainty[pc]"][0]
        reddening_uncertainty_min_stilism = dfstilism["reddening_uncertainty_min[mag]"][0]
        reddening_uncertainty_max_stilism = dfstilism["reddening_uncertainty_max[mag]"][0]

        # Calculate the mean reddening uncertainty
        reddening_uncertainty_mean = (reddening_uncertainty_min_stilism + reddening_uncertainty_max_stilism) / 2

    return distance_stilism, distance_uncertainty_stilism, reddening_stilism, reddening_uncertainty_mean



def GAIA_extinction_correction(G_in, bp_rp_in, EBV):
    """
    Determines the extinctions in different Gaia bands using the intrinsic G mag, BP-RP color, and the E(B-V) reddening
    """

    # Coefficients for extinction correction calculations
    c_Gm = [0.97610, -0.17040, 0.00860, 0.00110, -0.04380, 0.00130, 0.00990]
    c_BP = [1.15170, -0.08710, -0.03330, 0.01730, -0.02300, 0.00060, 0.00430]
    c_RP = [0.61040, -0.01700, -0.00260, -0.00170, -0.00780, 0.00005, 0.00060]

    # Calculate total extinction A0 based on the E(B-V) reddening
    A0 = 3.1 * EBV

    # Calculate extinction coefficients for G, BP, and RP bands
    kGm = c_Gm[0] + c_Gm[1] * bp_rp_in + c_Gm[2] * bp_rp_in ** 2 + c_Gm[3] * bp_rp_in ** 3 + c_Gm[4] * A0 + c_Gm[5] * (A0 ** 2) + c_Gm[6] * A0 * bp_rp_in
    kBP = c_BP[0] + c_BP[1] * bp_rp_in + c_BP[2] * bp_rp_in ** 2 + c_BP[3] * bp_rp_in ** 3 + c_BP[4] * A0 + c_BP[5] * (A0 ** 2) + c_BP[6] * A0 * bp_rp_in
    kRP = c_RP[0] + c_RP[1] * bp_rp_in + c_RP[2] * bp_rp_in ** 2 + c_RP[3] * bp_rp_in ** 3 + c_RP[4] * A0 + c_RP[5] * (A0 ** 2) + c_RP[6] * A0 * bp_rp_in

    # Apply extinction correction to G and BP-RP values
    G = G_in + (kGm * A0)
    BR = bp_rp_in + (kBP - kRP) * A0

    # Return the corrected G magnitude, corrected BP-RP color, and extinction coefficients
    return G, BR, kGm, kBP, kRP



def Gmag_intrinsic(G_observed, BR_observed, EBV, BR_range):
    """
    Determines the intrinsic G and BP-RP based on the observed magnitude/colors and the reddening
    """

    # Create an array of BR (BP - RP) values around the observed one
    BR_array = np.arange(BR_observed - BR_range, BR_observed + BR_range, 0.001)

    # Calculate the observed values for Gmag, Bp-RP, and extinctions in different bands
    G_array, BR_array, kGm_array, kBP_array, kRP_array = GAIA_extinction_correction(G_observed, BR_array, EBV)

    # Find the index of the case when the BR is close to the BR_observed
    condition = (BR_array < BR_observed + 0.0008) & (BR_array > BR_observed - 0.0008)

    # Determine the Kg = Ag/A0, Kbp, and Krp values based on the condition
    Kg = kGm_array[np.where(condition)[0][0]]
    Kbp = kBP_array[np.where(condition)[0][0]]
    Krp = kRP_array[np.where(condition)[0][0]]

    # Calculate the intrinsic bp_rp based on the observed BR and extinction coefficients
    bp_rp_intrinsic = BR_observed - Kbp * 3.1 * EBV + Krp * 3.1 * EBV

    # Calculate the intrinsic Gmag based on the observed G and extinction coefficient
    G_intrinsic = G_observed - Kg * 3.1 * EBV

    # Return the intrinsic G magnitude and intrinsic bp_rp color
    return G_intrinsic, bp_rp_intrinsic



def Vmag_from_Gmag(Gmag, bp_rp):
    """
    Determines the V mag from Gaia's Gmag and BP-RP color
    """

    # Coefficients for the Vmag - Gmag relation from Montalto et al. (2021)
    c_Gmag = [-0.17276, 0.47885, -0.71953, 0.24374, -0.04458, 0.00317]

    # Calculate V mag from Gmag and BP-RP color using the relation
    V_from_G = Gmag - (c_Gmag[0] + c_Gmag[1] * bp_rp + c_Gmag[2] * bp_rp ** 2 + c_Gmag[3] * bp_rp ** 3 + c_Gmag[4] * bp_rp ** 4 + c_Gmag[5] * bp_rp ** 5)

    # Return the V mag
    return V_from_G



def PosNormal(mean, sigma):
    """
    Function for generating a positive normal distribution with a specified mean and sigma.
    The generated value is constrained to be between 0.003 and 1.5, which corresponds to
    0.01 < A0 < 5 mag: the range of calibration.
    """

    # Generate a random value from a normal distribution with the given mean and sigma
    x = np.random.normal(mean, sigma, 1)

    # Check if the generated value is within the desired range
    # If it is, return the value, otherwise generate a new value recursively
    return x if x >= 0.003 and x <= 1.5 else PosNormal(mean, sigma)


def Vmag_error_from_Gmag(Gmag, Gmag_err, bp, bp_err, rp, rp_err, l, b, distance):
    """
    Determines the V mag and its error by perturbing the input parameters in a Monte Carlo simulation.
    """

    # Calculate the reddening and its uncertainty using stilism_distance.py
    distance_stilism, distance_uncertainty_stilism, reddening_stilism, reddening_uncertainty_mean = extinction_with_stilism(l, b, distance)

    # Calculate the observed BP-RP color and its error
    bp_rp = bp - rp
    bp_rp_err = (bp_err**2 + rp_err**2)**0.5

    N = 1000
    Gmag_arr = np.empty((0, N))
    bp_rp_arr = np.empty((0, N))
    Vmag_arr = np.empty((0, N))

    # Perform the Monte Carlo simulation
    for i in range(N):
        # Perturb the observed Gmag and BP-RP values using their errors
        Gmag_tmp = np.random.normal(Gmag, Gmag_err, 1)
        BR_observed_tmp = np.random.normal(bp_rp, bp_rp_err, 1)

        # Generate a reddening value within the range using PosNormal function
        EBV_tmp = PosNormal(reddening_stilism, reddening_uncertainty_mean)

        # Calculate the intrinsic Gmag and intrinsic BP-RP values
        BR_range = (EBV_tmp + bp_rp_err) * 3
        G_intrinsic_tmp, bp_rp_intrinsic_tmp = Gmag_intrinsic(Gmag_tmp, BR_observed_tmp, EBV_tmp, BR_range)

        # Calculate the intrinsic V mag from the intrinsic G mag and BP-RP color
        V_mag_intrinsic_tmp = Vmag_from_Gmag(G_intrinsic_tmp, bp_rp_intrinsic_tmp)

        # Store the generated values in arrays
        Vmag_arr = np.append(Vmag_arr, V_mag_intrinsic_tmp)
        Gmag_arr = np.append(Gmag_arr, G_intrinsic_tmp)
        bp_rp_arr = np.append(bp_rp_arr, bp_rp_intrinsic_tmp)

    # Calculate the mean and standard deviation of the generated values
    G_intrinsic = np.round(np.mean(Gmag_arr), 3)
    G_intrinsic_err = np.round(np.std(Gmag_arr), 3)
    V_intrinsic = np.round(np.mean(Vmag_arr), 3)
    V_intrinsic_err = np.round((np.std(Vmag_arr)**2 + 0.02745**2)**0.5, 3)  # 0.02745 is the dispersion of the fit of the Vmag - Gmag relation in Montalto et al. (2021)
    bp_rp_intrinsic = np.round(np.mean(bp_rp_arr), 3)
    bp_rp_intrinsic_err = np.round(np.std(bp_rp_arr), 3)

    # Return the intrinsic V mag, its error, intrinsic G mag, its error, intrinsic BP-RP, its error, reddening, and reddening uncertainty
    return V_intrinsic, V_intrinsic_err, G_intrinsic, G_intrinsic_err, bp_rp_intrinsic, bp_rp_intrinsic_err, reddening_stilism, reddening_uncertainty_mean

def gaia_DR3_query(dr3_id):
    """
    Performs a search in the Gaia eDR3 database and provides the necessary parameters for the determination of Vintrinsic.
    """

    # Define the uncertainties for Gmag, BPmag, and RPmag
    sigmaG_0 = 0.0027553202
    sigmaBP_0 = 0.0027901700
    sigmaRP_0 = 0.0037793818

    # Query the Gaia database for object parameters using the provided DR3 ID
    query = f"SELECT * FROM gaiadr3.gaia_source WHERE source_id = '{dr3_id}'"

    j = Gaia.launch_job(query)
    gaia_DR3_results_target = j.get_results()

    # Retrieve the Gmag, BPmag, RPmag, and BP-RP values from the query results
    Gmag = np.round(gaia_DR3_results_target['phot_g_mean_mag'], 4)
    BPmag = np.round(gaia_DR3_results_target['phot_bp_mean_mag'], 4)
    RPmag = np.round(gaia_DR3_results_target['phot_rp_mean_mag'], 4)
    BP_RP = BPmag - RPmag

    # Calculate the uncertainties for Gmag, BPmag, RPmag, and BP-RP
    e_Gmag = np.round(((-2.5 / math.log(10) / gaia_DR3_results_target['phot_g_mean_flux_over_error'])**2 + sigmaG_0**2)**0.5, 4)
    e_BPmag = np.round(((-2.5 / math.log(10) / gaia_DR3_results_target['phot_bp_mean_flux_over_error'])**2 + sigmaBP_0**2)**0.5, 4)
    e_RPmag = np.round(((-2.5 / math.log(10) / gaia_DR3_results_target['phot_rp_mean_flux_over_error'])**2 + sigmaRP_0**2)**0.5, 4)
    e_BP_RP = (e_BPmag**2 + e_RPmag**2)**0.5

    # Retrieve the galactic longitude, latitude, and distance values
    l = gaia_DR3_results_target['l'].value[0]
    b = gaia_DR3_results_target['b'].value[0]
    dist = np.round(1000. / gaia_DR3_results_target['parallax'].value[0], 2)

    # Return the Gmag, its error, BPmag, its error, RPmag, its error, galactic longitude, latitude, and distance
    return Gmag[0], e_Gmag[0], BPmag[0], e_BPmag[0], RPmag[0], e_RPmag[0], l, b, dist


def Vin_from_gaia_DR3_query(dr3_id):
    """
    Main/final function to provide Vintrinsic and its error using all the other functions.
    """

    # Retrieve the necessary parameters from the Gaia eDR3 database query
    Gmag, e_Gmag, BPmag, e_BPmag, RPmag, e_RPmag, l, b, dist = gaia_DR3_query(dr3_id)

    # Calculate V_intrinsic, its error, G_intrinsic, its error, bp_rp_intrinsic, bp_rp_intrinsic_err, reddening_stilism, and reddening_uncertainty_mean
    V_intrinsic, V_intrinsic_err, G_intrinsic, G_intrinsic_err, bp_rp_intrinsic, bp_rp_intrinsic_err, reddening_stilism, reddening_uncertainty_mean = Vmag_error_from_Gmag(Gmag, e_Gmag, BPmag, e_BPmag, RPmag, e_RPmag, l, b, dist)

    # Print the calculated V_intrinsic and its error
    print(f'V={V_intrinsic:.2f}±{V_intrinsic_err:.2f}  G={G_intrinsic:.2f}±{G_intrinsic_err:.2f}')

    # Return the calculated V_intrinsic, its error, G_intrinsic, its error, bp_rp_intrinsic, bp_rp_intrinsic_err, reddening_stilism, and reddening_uncertainty_mean
    return V_intrinsic, V_intrinsic_err, G_intrinsic, G_intrinsic_err, bp_rp_intrinsic, bp_rp_intrinsic_err, reddening_stilism, reddening_uncertainty_mean


if __name__ == "__main__":
    Vin_from_gaia_DR3_query(dr3_id=str(sys.argv[1:][0]))
