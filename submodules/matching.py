""" Matching: select points from the detected which match the fitted surface.
"""

import numpy as np
import skimage
from etsynseg import imgutil, pcdutil
from etsynseg import features, dtvoting

__all__ = [
    "GMMFixed",
    "match_by_closeness",
    "bridge_gaps_2d", "bridge_gaps_3d",
    "match_candidate_to_ref"
]

#=========================
# GMM
#=========================

class GMMFixed:
    """ Gaussian mixture model (GMM) with fixed means.
    """
    def __init__(self, means_fixed=(0, np.pi/2),
        sigmas_init=(0.1, 0.1), weights_init=(0.5, 0.5),
        tol=0.001, max_iter=100,
        ):
        """ Initialize the GMM.
        
        Args:
            means_fixed (2-tuple): Fix means to this value, (mean_signal,mean_noise).
            sigmas_init (2-tuple): Initial (sigma_signal,sigma_noise).
            weights_init (2-tuple): Initial (weight_signal,weight_noise).
            tol (float): Tolerance for convergence.
            max_iter (int): Max number of iterations.
        """
        self.means = means_fixed
        self.sigmas = sigmas_init
        self.weights = weights_init
        self.tol = tol
        self.max_iter = max_iter

    def gauss(self, x, sigma):
        """ Gaussian function, e^(-x^2/2sigma^2)/(sqrt(2pi)*sigma)

        Args:
            x, sigma (float): Parameters of the function.
        """
        return np.exp(-x**2/(2*sigma**2))/((2*np.pi)**0.5*sigma)

    def posterior(self, x):
        """ Posterior calculation, E-step.

        Args:
            x (np.ndarray): 1d array for data.

        Returns:
            p_post (np.ndarray): Shape=(2,len(x)). p_post[i] is the posterior for category i (0 for signal, 1 for noise).
        """
        # conditional probabilities
        p_cond = np.empty((2, len(x)))
        for i in range(2):
            p_cond[i] = self.weights[i]*self.gauss((x-self.means[i]), self.sigmas[i])

        # posterior probabilities
        p_post_sum = np.sum(p_cond, axis=0)
        p_post = np.empty((2, len(x)))
        for i in range(2):
            p_post[i] = p_cond[i] / p_post_sum
        return p_post

    def maximize_params(self, x, p_post):
        """ Maximize parameters, M-step.

        Args:
            x (np.ndarray): 1d array for data.
            p_post (np.ndarray): Posterior with shape=(2,len(x)).

        Returns:
            weights (np.ndarray): Shape=(2,len(x)). Weight parameters for (signal,noise).
            sigmas (np.ndarray): Shape=(2,len(x)). Sigma parameters for (signal,noise).
        """
        weights = tuple(np.mean(p_post[i]) for i in range(2))
        sigmas = tuple(
            np.sqrt(np.sum(p_post[i]*(x-self.means[i])**2)/np.sum(p_post[i]))
            for i in range(2)
        )
        return weights, sigmas

    def fit_threshold(self, x):
        """ Fit data. Calculate a threshold value that separates signal and noise.

        Args:
            x (np.ndarray): 1d array for data.

        Returns:
            x_thresh (float): Considered x<x_thresh as signals.
        """
        for i in range(self.max_iter):
            # E-step
            if i == 0:
                p_post = self.posterior(x)
            else:
                p_post = p_post_new

            # M-step
            self.weights, self.sigmas = self.maximize_params(x, p_post)

            # termination
            p_post_new = self.posterior(x)
            if np.max(p_post_new-p_post) < self.tol:
                break

        # get threshold
        p_post = self.posterior(x)
        x_thresh = np.max(x[p_post[0] > 0.5])
        return x_thresh


#=========================
# match by spatial and orientational closeness
#=========================

def match_by_closeness(Bcand, Ocand, Bref, Oref, sigma_tv):
    """ Match the candidate image to the reference one by spatial and orientational closeness.
    
    First apply hessian+TV to the reference surface to extend it to nearby regions.
    Compare the orientations between the extended reference and the candidate, and threshold by GMM.
    Raw matched pixels in the candidate image are spatially and orientationally close to the reference.
    
    Args:
        Bcand, Ocand (np.ndarray): Candidate binary image and orientation, with shape=(nz,ny,nx).
        Bref, Oref (np.ndarray): Reference binary image and orientation, with shape=(nz,ny,nx).
        sigma_tv (float): Sigma for TV of the reference surface.
    
    Returns:
        Bmatch (np.ndarray): Matched binary image, with shape=(nz,ny,nx).
    """
    # spatial closeness: TV for fit
    Sfit_tv, Oref_tv = dtvoting.stick3d(Bref, Oref*Bref, sigma=sigma_tv)

    # mask: belongs to B and close to Bfit
    # e^(-1/2) = e^(-r^2/(2*sigma_tv^2)) at r=sigma_tv, close to fill holes
    Bmask = Sfit_tv > np.exp(-1/2)
    Bmask = skimage.morphology.binary_closing(Bmask)
    Bmask = (Bmask*Bcand).astype(bool)

    # difference in orientation
    dO = imgutil.orients_absdiff(Ocand, Oref_tv)

    # estimate dO threshold using GMM
    gmm = GMMFixed(means_fixed=(0, np.pi/2))
    dO_thresh = gmm.fit_threshold(dO[Bmask])

    # match by threshold
    Bmatch = Bmask * (dO < dO_thresh)

    return Bmatch

def bridge_gaps_2d(pts_cand, pts_gapped, orients_cand, r_thresh):
    """ Bridge gaps for a 2d slice.

    Gapped points is a subset of candidate points.
    If separate components in gapped points can be connected in the candidate
    (both in the sense of neighbors graph),
    then build the bridge as the shortest path in the connection.

    Args:
        pts_cand (np.ndarray): Candidate points, with shape=(npts_cand,dim). Assumed sorted.
        pts_gapped (np.ndarray): Gapped points, with shape=(npts_gapped,dim).
        orients_cand (np.ndarray): Orientation of each point in candidate, with shape=(npts_cand,).
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.

    Returns:
        pts_bridge (np.ndarray or None): Bridged points, with shape=(npts_bridged,dim).
            Return None if no points are matched.
    """
    # build neighbors graph for candidate points
    g_cand = pcdutil.neighbors_graph(
        pts_cand, r_thresh=r_thresh, orients=orients_cand
    )
    # mask for candidate points that are also gapped points (by dist=0)
    mask_ingapped = np.isclose(
        pcdutil.points_distance(pts_cand, pts_gapped, return_2to1=False),
        0
    )

    # iterate for each component in g_cand
    pts_bridge = []
    membership = np.asarray(g_cand.components(mode="weak").membership)
    for i in np.unique(membership):
        # index of points in the current component that are also part of gapped points
        mask_comp_i = (membership == i)
        idx_comp_ingapped = np.nonzero(mask_comp_i & mask_ingapped)[0]
        
        # path from the 1st occurrence of gapped points in the component to the last
        # this path covers the middle part in candidate points, thus a bridge
        if len(idx_comp_ingapped) > 0:
            idx_path = g_cand.get_shortest_paths(
                min(idx_comp_ingapped), max(idx_comp_ingapped),
                weights=g_cand.es["dorients"]
            )[0]
            pts_bridge.append(pts_cand[idx_path])
    
    # concat
    if len(pts_bridge) > 0:
        pts_bridge = np.concatenate(pts_bridge, axis=0)
        return pts_bridge
    else:
        return None

def bridge_gaps_3d(zyx_cand, zyx_gapped, orients_cand, r_thresh):
    """ Bridge gaps slice by slice.

    Gapped points is a subset of candidate points.
    For each slice, if separate components in gapped points can be connected in the candidate
    (both in the sense of neighbors graph),
    then build the bridge as the shortest path in the connection.

    Args:
        zyx_cand (np.ndarray): Candidate points, with shape=(npts_cand,3). Assumed sorted.
        zyx_gapped (np.ndarray): Gapped points, with shape=(npts_gapped,3).
        orients_cand (np.ndarray): Orientation of each point in candidate, with shape=(npts_cand,).
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.

    Returns:
        zyx_bridge (np.ndarray): Bridged points, with shape=(npts_bridged,3).
    """
    # obtain bridge points slice by slice
    zyx_bridged = []
    for z in np.unique(zyx_cand[:, 0]):
        # extract points in slice-z
        mask_cand_z = zyx_cand[:, 0] == z
        zyx_cand_z = zyx_cand[mask_cand_z]
        orients_cand_z = orients_cand[mask_cand_z]
        zyx_gapped_z = zyx_gapped[zyx_gapped[:, 0] == z]

        # bridge gap
        zyx_bridged_z = bridge_gaps_2d(
            zyx_cand_z, zyx_gapped_z,
            orients_cand=orients_cand_z,
            r_thresh=r_thresh
        )
        if zyx_bridged_z is not None:
            zyx_bridged.append(zyx_bridged_z)

    # concat bridged points
    zyx_bridged = np.concatenate(zyx_bridged, axis=0)
    return zyx_bridged
    
def match_candidate_to_ref(zyx_cand, zyx_ref, guide, r_thresh):
    """ Match the candidate image to the reference one.
    
    First match by spatial and orientational closeness.
    Then bridge the gaps.
    The result is a subset of candidate points that match the reference.

    Args:
        zyx_cand (np.ndarray): Candidate points, with shape=(npts_cand,3). Assumed sorted.
        zyx_ref (np.ndarray): Reference points, with shape=(npts_gapped,3).
        guide (np.ndarray): Guideline points which are sorted, with shape=(npts_guide,3).
        r_thresh (float): Distance threshold for point pairs to be counted as neighbors.

    Returns:
        zyx_match (np.ndarray): Matched points, with shape=(npts_match,3). Sorted by guide.
    """
    # match by spatial-orientational closeness
    
    # setup shape, add margin in high boundary
    _, range_high, _ = pcdutil.points_range(
        np.concatenate([zyx_cand, zyx_ref], axis=0)
    )
    margin_high = np.ceil([1, 2*r_thresh, 2*r_thresh])
    shape = (range_high + margin_high).astype(int)


    # convert points to pixels
    Bcand = pcdutil.points2pixels(zyx_cand, shape)
    _, Ocand = features.ridgelike3d(Bcand, sigma=r_thresh)
    Bref = pcdutil.points2pixels(zyx_ref, shape)
    _, Oref = features.ridgelike3d(Bref, sigma=r_thresh)
    
    # match
    Bclose = match_by_closeness(Bcand, Ocand, Bref, Oref, sigma_tv=r_thresh)
    zyx_close = pcdutil.pixels2points(Bclose)


    # bridge gaps

    # sort points
    zyx_cand = pcdutil.sort_pts_by_guide_3d(zyx_cand, guide)
    # extract orientation
    orients_cand = Ocand[tuple(zyx_cand.T)]
    
    # bridge
    zyx_match = bridge_gaps_3d(
        zyx_cand, zyx_close, orients_cand, r_thresh=r_thresh
    )
    # sort
    zyx_match = pcdutil.sort_pts_by_guide_3d(zyx_match, guide)
    return zyx_match
