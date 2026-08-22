"""
Custom implementation of the Pyramidal Lucas-Kanade optical flow algorithm.
"""
import numpy as np

import cv2


def min_eigenvalue2x2(A: np.ndarray, n_pixels: int) -> float:
    """
    Computes the minimum eigenvalue of a 2x2 matrix normalized by the number of pixels and the gradient gain.

    Normalization notes:
    - The `n_pixels` makes the independent of the patch size.

    Args:
        A: 2x2 matrix.
        n_pixels: Number of pixels in the patch.

    Returns:
        Minimum eigenvalue.
    """
    trace = A[0, 0] + A[1, 1]
    determinant = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    discriminant = max(trace**2 - 4 * determinant, 0)
    eig = (trace - np.sqrt(discriminant)) / 2
    return eig / n_pixels


def solve_2x2_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Exploiting the fact that the system is 2x2 and the matrix A is constant for each iteration. 
    This leads to a faster solution than the general case (~1.2x faster).

    Args:
        A: 2x2 matrix.
        b: 2x1 vector.

    Returns:
        Solution of the 2x2 system.
    """
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    dx = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
    dy = (b[1] * A[0, 0] - b[0] * A[1, 0]) / det
    return np.array([dx, dy])


class PyLucasKanadeEstimator:
    """
    Custom implementation of the Pyramidal Lucas-Kanade optical flow algorithm. 
    Inspired by the implementation in the `opencv` library.
    Reference: https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html

    Algorithm (pseudo-code):
    ```
    for each pyramid level, coarse -> fine:

        for each feature:

            p = feature position in frame t

            Initialize flow:
            - 0 at the coarsest level (start as there is not motion)
            - 2 * previous flow otherwise

            Sample patch around p in frame t

            Compute Ix, Iy and the 2x2 LK matrix G

            for each LK iteration:

                q = p + flow

                Sample patch around q in frame t+1

                Compute residual:
                It = patch_q - patch_p

                Solve the 2x2 system for correction:
                d = (du, dv)

                Update:
                flow += d

                Stop if:
                    norm(d) < epsilon

            Save tracked point:
                dst = p + flow
    ```
    Notes: 
    - The lstsq is implicitly performed exploiting the fact that the system is 2x2 and the matrix G is constant for each iteration (as the gradients are calculated over the same patch).
    - Lucas-Kanade works with grayscale images.

    Least squares solution:
    ```
    G = np.array([  // gradient matrix
        [np.sum(Ix * Ix), np.sum(Ix * Iy)],
        [np.sum(Ix * Iy), np.sum(Iy * Iy)],
    ])

    b = -np.array([  // intensity residual vector
        np.sum(Ix * It),
        np.sum(Iy * It),
    ])

    d = solve_2x2_system(G, b)  // least squares solution

    du, dv = d  // flow vector
    ```
    """
    def __init__(
        self,
        window_size: int | tuple[int, int] = 21,
        max_level: int = 3,
        max_iterations: int = 30,
        iteration_convergence_threshold: float = 0.1,
        resize_interpolation_algorithm: int = cv2.INTER_NEAREST,
        min_eigenvalue_threshold: float = 1e-3,
        gradient_gain: float = 32.0,
        implementation: str = 'custom'
    ):
        """
        Args:
            window_size: Size of the search window.
            max_level: Maximum number of pyramid levels.
            max_iterations: Maximum number of iterations.
            iteration_convergence_threshold: Threshold for convergence.
            resize_interpolation_algorithm: Interpolation algorithm to use for resizing the frame to the given level.
            min_eigenvalue_threshold: Threshold for the minimum eigenvalue of the gradient matrix.
            gradient_gain: Gain for the gradient calculation.
            implementation: `custom` runs the code below, `opencv` delegates to
                `cv2.calcOpticalFlowPyrLK`. The two agree to 0.004 px (Appendix B.1); the OpenCV
                path exists so that a throughput comparison is not charged for this being Python.
        """
        assert implementation in ('custom', 'opencv'), \
            f'Unknown implementation "{implementation}", expected "custom" or "opencv"!'
        self._window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self._max_level = max_level
        self._max_iterations = max_iterations
        self._iteration_convergence_threshold = iteration_convergence_threshold
        self._resize_interpolation_algorithm = resize_interpolation_algorithm
        self._min_eigenvalue_threshold = min_eigenvalue_threshold
        self._gradient_gain = gradient_gain
        self._implementation = implementation

    def estimate(self, prev_frame: np.ndarray, next_frame: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates the optical flow between the previous and next frames. Check the class documentation for the algorithm details.

        Args:
            prev_frame: Previous frame.
            next_frame: Next frame.
            points: Points to estimate the flow for.

        Important:
        - It is assumed that the points are absolute (not normalized).

        Returns:
            Optical flows for each point and a boolean array indicating if the point is valid for tracking.
        """
        if self._implementation == 'opencv':
            return self._estimate_opencv(prev_frame, next_frame, points)

        if prev_frame.shape != next_frame.shape:
            raise ValueError("The previous and next frames must have the same shape.")

        # Preprocess the frames (RGB -> grayscale if needed)
        if prev_frame.ndim == 3 and prev_frame.shape[2] == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray, next_gray = prev_frame, next_frame

        # State
        flows = np.zeros((len(points), 2), dtype=np.float32)  # Initialize with flows to zeros (start as there is not motion)
        skips = np.zeros((len(points),), dtype=bool)

        # Algorithm
        levels = list(range(self._max_level - 1, -1, -1))
        for level_idx, level in enumerate(levels):
            prev_level_gray = self._scale_frame_to_level(prev_gray, level)
            next_level_gray = self._scale_frame_to_level(next_gray, level)

            # Pre-computation (avoid computing gradients per point) - faster in cases there are many points
            Ix, Iy = self._compute_gradients(prev_level_gray)

            for point_idx, point in enumerate(points):
                if skips[point_idx]:
                    continue

                flow_prior = flows[point_idx, :] * (1 if level_idx == 0 else 2)  # Flow initialization for the current level
                flow, valid = self._compute_flow(
                    prev_level_frame=prev_level_gray, 
                    next_level_frame=next_level_gray, 
                    prev_grad_x=Ix, 
                    prev_grad_y=Iy, 
                    point=point, 
                    level=level, 
                    flow_prior=flow_prior
                )  # compute the flow for the current point
                if not valid:
                    skips[point_idx] = True
                    continue

                flows[point_idx, :] = flow  # update the flow for the current point

        return flows, skips

    def _estimate_opencv(self, prev_frame: np.ndarray, next_frame: np.ndarray,
                         points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        The same estimate from `cv2.calcOpticalFlowPyrLK`, in this class's return convention.

        OpenCV returns the tracked positions and a status flag; this class returns flows and a
        skip mask, so the positions are differenced and the flag inverted. The termination
        criteria and window are taken from this instance so both paths run the same settings.
        """
        if prev_frame.ndim == 3 and prev_frame.shape[2] == 3:
            previous = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            current = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        else:
            previous, current = prev_frame, next_frame
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, points.astype(np.float32).reshape(-1, 1, 2), None,
            winSize=self._window_size,
            maxLevel=self._max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      self._max_iterations, self._iteration_convergence_threshold),
            minEigThreshold=self._min_eigenvalue_threshold
        )
        flows = (tracked.reshape(-1, 2) - points).astype(np.float32)
        skips = status.reshape(-1) == 0
        flows[skips] = 0.0
        return flows, skips

    def _scale_frame_to_level(self, frame: np.ndarray, level: int) -> np.ndarray:
        """
        Scales the frame to the given level (pyramid level).

        Args:
            frame: Frame to scale.
            level: Pyramid level.

        Returns:
            Scaled frame.
        """
        if level == 0:
            return frame

        scale_factor = 2**level
        level_height, level_width = frame.shape[0] // scale_factor, frame.shape[1] // scale_factor
        return cv2.resize(frame, (level_width, level_height), interpolation=self._resize_interpolation_algorithm)  

    def _compute_flow(
        self, 
        prev_level_frame: np.ndarray, 
        next_level_frame: np.ndarray,
        prev_grad_x: np.ndarray,
        prev_grad_y: np.ndarray,
        point: np.ndarray, 
        level: int, 
        flow_prior: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """
        Computes the optical flow for the given point.

        Args:
            prev_level_frame: Previous grayscale frame, scaled to the level.
            next_level_frame: Next grayscale frame, scaled to the level.
            prev_grad_x: Previous gradient x.
            prev_grad_y: Previous gradient y.
            point: Point to compute the flow for.
            level: Level of the pyramid.
            flow_prior: Flow prior.

        Returns:
            Flow delta and if the point is feature is valid for tracking
        """
        flow = flow_prior
        scale_factor = 2**level
        patch_p = self._sample_patch(prev_level_frame, point, level).astype(np.float32)
        Ix, Iy = self._sample_gradients(prev_grad_x, prev_grad_y, point, level)
        Ix, Iy = Ix.ravel(), Iy.ravel()  # Faster to perform dot product instead of np.sum(a, b) - avoid unnecessary memory allocations
        G = np.array([
            [Ix @ Ix, Ix @ Iy],
            [Ix @ Iy, Iy @ Iy],
        ])
        if min_eigenvalue2x2(G, patch_p.size) < self._min_eigenvalue_threshold:
            # The gradient matrix is singular, so the point/feature cannot be tracked
            return flow, False

        for _ in range(self._max_iterations):
            q = point + flow * scale_factor  # the `p` and `q` have level independent coordinates, but the `flow` has level dependent coordinates (so it needs to be scaled)
            patch_q = self._sample_patch(next_level_frame, q, level).astype(np.float32)
            It = (patch_q - patch_p).ravel()

            b = -np.array([Ix @ It, Iy @ It])
            d = solve_2x2_system(G, b)
            flow += d

            if np.linalg.norm(d) < self._iteration_convergence_threshold:
                break

        return flow, True

    def _sample_patch(
        self, 
        gray_frame: np.ndarray, 
        point: np.ndarray,
        level: int,
    ) -> np.ndarray:
        """
        Samples the patch around the given point.

        Args:
            gray_frame: Grayscale frame to sample the patch from.
            point: Point to sample the patch around.
            level: Level of the pyramid.

        Returns:
            Sampled patch.
        """
        scale_factor = 2**level
        x, y = point[0] / scale_factor, point[1] / scale_factor
        patch_size = (self._window_size[1], self._window_size[0])
        return cv2.getRectSubPix(gray_frame, patch_size, (x, y))

    def _sample_gradients(
        self,
        prev_grad_x: np.ndarray,
        prev_grad_y: np.ndarray,
        p: np.ndarray,
        level: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples the gradients around the given point.

        Args:
            prev_grad_x: Previous gradient x.
            prev_grad_y: Previous gradient y.
            p: Point to sample the gradients around.

        Returns:
            Sampled gradients.
        """
        scale_factor = 2**level
        x, y = p[0] / scale_factor, p[1] / scale_factor
        patch_size = (self._window_size[1], self._window_size[0])
        Ix = cv2.getRectSubPix(prev_grad_x, patch_size, (x, y))
        Iy = cv2.getRectSubPix(prev_grad_y, patch_size, (x, y))
        return Ix, Iy

    def _compute_gradients(self, patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the gradients of the given patch.

        Args:
            patch: Patch to compute the gradients of.

        Returns:
            Gradients.
        """
        Ix = cv2.Scharr(patch, cv2.CV_32F, 1, 0, scale=1 / self._gradient_gain)
        Iy = cv2.Scharr(patch, cv2.CV_32F, 0, 1, scale=1 / self._gradient_gain)
        return Ix, Iy
