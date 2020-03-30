from __future__ import division, print_function
import farey
import radon
import finite
import numbertheory as nt
import numpy as np
import scipy.fftpack as fftpack
import pyfftw
import math


# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft


class Sampler(object):

    def __init__(self, seed=0, verbose=False):
        self.seed = seed
        self.verbose = verbose
        self.mask = None
        self.r_no_tile = None
        self.r_actual = None

    def reduction_factor(self, mask=None):
        if mask is None:
            if self.mask is None:
                raise AttributeError("Mask not generated yet")
            else:
                mask = self.mask
        sampled_points = np.count_nonzero(mask)
        total_points =  mask.size
        return total_points / float(sampled_points)

    def generate_mask(self, size):
        raise NotImplementedError("Method must be defined in subclass.")

    def sample_kspace(self, image):
        if self.mask is None:
            self.generate_mask(image.shape[:-2])
        kspace_input = fftpack.fft2(image.astype(np.complex64))
        kspace_sampled = kspace_input * self.mask
        image_sampled = fftpack.ifft2(kspace_sampled)
        image_sampled_real = np.real(image_sampled)[..., np.newaxis]
        image_sampled_imag = np.imag(image_sampled)[..., np.newaxis]
        image_sampled_cat = np.concatenate([image_sampled_real, image_sampled_imag], axis=2)
        return image_sampled_cat.astype(np.float32)

    def tile_center(self, size, mask, radius):
        center_mask = np.zeros(size, np.float)
        count = 0
        if radius <= 0:
            return count, mask, center_mask
        centerX = size[0]/2
        centerY = size[1]/2
        for i, row in enumerate(mask):
            for j, col in enumerate(row):
                distance = math.sqrt( (i-float(centerX))**2 + (j-float(centerY))**2)
                if distance < radius:
                    if not mask[i, j] > 0:  # already selected
                        count += 1
                        mask[i, j] = 1
                        center_mask[i, j] = 1
        return count, mask, center_mask

    def do_transform(self, x):
        x = self.sample_kspace(x)
        return x

    def vprint(self, *args):
        if self.verbose: print(*args)


class OneDimCartesianRandomSampler(Sampler):
    """
    Transform the image to Fourier space, sample the array
    along randomly-spaced axis-aligned lines, then convert
    the result back to image space.
    
    Terminology comes from the MRI domain, where this method has relevance.
    """
    
    def __init__(self, r=5.0, r_alpha=3, axis=1, acs=3, seed=0):
        """
        Arguments:
            r: float. Reduction factor. A 1/r fraction of k-space will be sampled,
                excluding the auto-calibration signal region.
            r_alpha: float. Higher values mean lower-frequency samples are more likely.
            axis: int. The axis the sampling lines are aligned to.
            acs: int. Width of the auto-calibration signal region, 
                which is fully-sampled about the center of k-space.
            seed: int. Random seed. A positive number gives repeatable "randomness".
        """
        super(OneDimCartesianRandomSampler, self).__init__(seed)
        self.r,self.r_alpha,self.axis,self.acs = r,r_alpha,axis,acs
        self.index_sample = None

    def tile(self, size, index_sample):
        mask = np.zeros(size)
        # Set the samples in the mask
        if self.axis == 0:
            mask[index_sample, :] = 1
        else:
            mask[:, index_sample] = 1
        self.r_no_tile = len(mask.flatten()) / np.sum(mask.flatten())
        # ACS
        mask = self.tile_acs(mask)
        # Compute reduction
        self.r_actual = len(mask.flatten()) / np.sum(mask.flatten())
        mask = mask.astype(np.bool)
        mask = fftpack.fftshift(mask)
        self.mask = mask
        return mask

    def tile_acs(self, mask):
        if self.acs <= 1:
            return mask
        acs1 = int((self.acs + 1) / 2)
        acs2 = -int(self.acs / 2)
        if self.axis == 0:
            mask[:acs1, :] = 1
            if self.acs > 1: mask[acs2:, :] = 1
        else:
            mask[:, :acs1] = 1
            if self.acs > 1: mask[:, acs2:] = 1
        return mask

    def generate_mask(self, size):
        """
        Generates a mask array for variable-density cartesian 1D sampling.
        Sampling probability is proportional to the position along the sampling axis,
        such that lower frequencies are more likely. The alpha value controls the 
        proportionality, e.g. alpha = 1 is linear, alpha = 2 is square.
        """
        # Initialise
        if self.seed >= 0:
            np.random.seed(self.seed)
        if type(size) != tuple or len(size) != 2:
            raise ValueError("Size must be a 2-tuple of ints")
        # Get sample coordinates
        num_phase_encode = size[self.axis]
        num_phase_sampled = int(np.floor(num_phase_encode / self.r)) - self.acs
        if num_phase_sampled < 0:
            raise ValueError("ACS is too large for the reduction factor")
        coordinate_normalized = np.arange(num_phase_encode)
        coordinate_normalized = np.abs(coordinate_normalized - num_phase_encode/2) \
                                / (num_phase_encode/2.0)
        prob_sample = coordinate_normalized**self.r_alpha
        if self.acs > 0:
            acs1 = int((self.acs + 1) / 2)
            acs2 = -int(self.acs / 2)
            prob_sample[:acs1] = 0
            if self.acs > 1: prob_sample[acs2:] = 0
        prob_sample = prob_sample / np.sum(prob_sample)
        self.index_sample = np.random.choice(num_phase_encode, size=num_phase_sampled, 
                                        replace=False, p=prob_sample)
        return self.tile(size, self.index_sample)

    
class TwoDimCartesianRandomSampler(Sampler):
    """
    Transform the image to Fourier space, sample the array
    at random points, then convert the result back to image space.
    
    Terminology comes from the MRI domain, where this method has relevance.
    """

    def __init__(self, r=5.0, r_alpha=3, acs=3, seed=0, verbose=False):
        """
        Arguments:
            r: float. Reduction factor. A 1/r fraction of k-space will be sampled,
                excluding the auto-calibration signal region.
            r_alpha: float. Higher values mean lower-frequency samples are more likely.
            axis: int. The axis the sampling lines are aligned to.
            acs: int. Width of the auto-calibration signal region, 
                which is fully-sampled about the center of k-space.
            seed: int. Random seed. A positive number gives repeatable "randomness".
        """
        super(TwoDimCartesianRandomSampler, self).__init__(seed, verbose)
        self.r, self.r_alpha, self.acs = r, r_alpha, acs

    def get_index_sample(self, size):
        num_phase_encode = size[0]  # assumes square
        _, _, center_mask = self.tile_center(size, np.zeros(size), self.acs)
        center_mask = fftpack.fftshift(center_mask)
        center_mask_inv = np.invert(center_mask.astype(np.bool))
        num_phase_sampled = int(np.floor(num_phase_encode**2 / self.r) - np.sum(center_mask))
        if num_phase_sampled < 1:
            raise ValueError("ACS is too large for the reduction factor")
        coordinate_normalized = np.arange(num_phase_encode)
        coordinate_normalized = np.abs(coordinate_normalized - num_phase_encode/2) \
                                / (num_phase_encode/2.0)
        prob_sample = coordinate_normalized**self.r_alpha
        prob_sample_2d = prob_sample[:, np.newaxis] * prob_sample[np.newaxis, :]
        prob_sample_2d *= center_mask_inv
        prob_sample = prob_sample_2d.flatten()
        prob_sample = prob_sample / np.sum(prob_sample)
        return np.random.choice(num_phase_encode**2, size=num_phase_sampled, 
                                replace=False, p=prob_sample)

    def tile(self, size, index_sample):
        mask = np.zeros(size)
        # Set the samples in the mask
        mask = mask.flatten()
        mask[index_sample] = 1
        mask = mask.reshape(size)
        self.r_no_tile = len(mask.flatten()) / np.sum(mask.flatten())
        # ACS
        mask = fftpack.fftshift(mask)
        self.tile_center(size, mask, self.acs)
        # Compute reduction
        self.r_actual = len(mask.flatten()) / np.sum(mask.flatten())
        mask = mask.astype(np.bool)
        self.mask = mask
        return mask

    def generate_mask(self, size):
        """
        Generates a mask array for variable-density cartesian 1D sampling.
        Sampling probability is proportional to the position along the sampling axis,
        such that lower frequencies are more likely. The alpha value controls the 
        proportionality, e.g. alpha = 1 is linear, alpha = 2 is square.
        """
        # Initialise
        if self.seed >= 0:
            np.random.seed(self.seed)
        if type(size) != tuple or len(size) != 2:
            raise ValueError("Size must be a 2-tuple of ints")
        # Get sample coordinates
        self.index_sample = self.get_index_sample(size)
        return self.tile(size, self.index_sample)
    

class FractalSampler(Sampler):
    """
    Transform the image to Fourier space, sample the array
    in a partially random fractal pattern composed of angled lines, 
    then convert the result back to image space.
    
    Terminology comes from the MRI domain, where this method has relevance.
    """

    REDUCTION_R_CTR_0  = {2: 0.50, 4: 0.25, 8: 0.125}
    REDUCTION_R_CTR_12 = {2: 0.49, 4: 0.23, 8: 0.11}
    REDUCTION_R_CTR_8  = {2: 0.47, 4: 0.21, 8: 0.08}
    REDUCTION_K_CTR_0  = {2: 2.40, 4: 0.85, 8: 0.29}
    REDUCTION_K_CTR_12 = {2: 2.30, 4: 0.75, 8: 0.24}
    REDUCTION_K_CTR_8  = {2: 2.20, 4: 0.65, 8: 0.14}
    PSEUDORANDOM_K = 0.1
    
    def __init__(self, k=1, K=0.1, r=0.48, ctr=1/12, subsets=8, 
            centered=True, two_quads=True, seed=0, verbose=False):
        """
        Arguments:
            k: float. 
            K: float. Relates to the Katz criterion.
                Indirectly controls the reduction factor.
            r: float. Controls the reduction factor for partially random fractals.
            two_quads: bool. If True, generate separate patterns in two quadrants
                instead of one.
            seed: int. Random seed. Non-negative int for repeatable pseudo-randomness.
        """
        super(FractalSampler, self).__init__(seed, verbose)
        self.k, self.K, self.r, self.ctr, self.subsets =  k, K, r, ctr, subsets
        self.centered, self.two_quads, self.seed = centered, two_quads, seed
        self.lines_set, self.angles_set, self.m_values = None, None, None

    def compute_sets(self, N, dummy_img):
        fareyVectors = farey.Farey()
        fareyVectors.compactOn()
        fareyVectors.generateFiniteWithCoverage(N)
        # Sort to reorder result for prettier printing
        finiteAnglesSorted, anglesSorted = fareyVectors.sort('Euclidean')
        self.vprint("Finite Coverage mu:", len(finiteAnglesSorted))

        if self.subsets:
            return finite.computeRandomLinesSubsets(
                    self.subsets, dummy_img, anglesSorted, finiteAnglesSorted,
                    self.r, self.K, self.centered, self.two_quads)
        else:
            return finite.computeRandomLines(
                    dummy_img, anglesSorted, finiteAnglesSorted, 
                    self.r, self.K, self.centered, self.two_quads)

    def tile_fractal(self, M, lines_set, angles_set, m_values):
        if self.subsets:
            mu = sum([len(lines) for lines in lines_set])
        else:
            mu = len(lines_set)
        total_samples = mu*(M-1) + 1
        self.vprint("Number of finite lines:", mu)
        self.vprint("Number of finite points:", total_samples)
        self.vprint("Angles set:", angles_set)

        # Set the samples in the mask from along the lines
        mask = np.zeros((M,M), np.float)
        if self.subsets:
            for lines in lines_set:
                self.tile_lines(M, mask, lines)
        else:
            self.tile_lines(M, mask, lines_set)
        mask[mask > 1] = 1
        self.r_no_tile = self.reduction_factor(mask)

        # Tile center region further
        count, _, center_mask = self.tile_center((M, M), mask, self.ctr*M)
        total_samples += count

        mask[mask > 1] = 1
        self.r_actual = self.reduction_factor(mask)
        self.vprint("Number of finite points (center tiled): ", total_samples)
        self.vprint("Reduction Factor: ", self.r_actual)

        mask = mask.astype(np.bool)
        self.mask = mask

    def tile_lines(self, M, mask, lines):
        for line in lines:
            u, v = line
            for x, y in zip(u, v):
                x %= M
                y %= M
                mask[x, y] += 1
        
    def generate_mask(self, size):
        """
        Generates a mask array for fractal sampling.
        The fractal is derived from Farey vectors and composed of straight
        lines at varied angles.
        Some lines may be randomly configured to introduce incoherence in 
        the resulting artefacts.
        """
        # Initialise
        seed = self.seed if self.seed >= 0 else None
        np.random.seed(seed)
        # Generate lines in fractal
        N = size[0]
        M = self.k * N
        p = nt.nearestPrime(M)

        dummy_img = np.zeros((p, p), dtype=np.float64)
        self.lines_set, self.angles_set, self.m_values = self.compute_sets(N, dummy_img)
        self.tile_fractal(M, self.lines_set, self.angles_set, self.m_values)

        return self.mask

if __name__ == '__main__':
    # sampler = OneDimCartesianRandomSampler(r=8.0, r_alpha=2, acs=32, seed=-1)
    # sampler = TwoDimCartesianRandomSampler(r=4.0, r_alpha=2, acs=32, seed=-1)
    # sampler = FractalSampler(ctr=0., K=0.1, r=0.23, centered=True)
    sampler = FractalSampler(K=0.14, r=0., ctr=1/8., centered=True)
    for K in np.arange(0.14, 0.20, 0.005):
        print(K)
        sampler.K = K
        sampler.generate_mask((256, 256))
        print(sampler.r_actual)
    # import matplotlib.pyplot as plt
    # plt.gray()
    # plt.imshow(sampler.mask)
    # plt.show()
