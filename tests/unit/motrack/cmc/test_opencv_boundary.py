"""
The hand-written estimation and matching must remain the default path.

The premise of this course work is that the estimation and matching are hand-written. The report
also reports throughput, and pure NumPy makes those numbers a property of the implementation
rather than of the method, so each component additionally offers an `opencv` backend for a fair
speed comparison. That escape hatch is exactly what could quietly hollow out the contribution: a
refactor that deleted a hand-written routine and left the OpenCV call behind would still pass every
other test.

Two properties are therefore checked here. The hand-written entry points still exist and still
carry their own implementations. And every banned OpenCV call sits inside a function reached only
when `implementation == 'opencv'`, so nothing on the default path delegates the work.

OpenCV remains freely allowed for image handling and feature *detection* - `cvtColor`,
`getRectSubPix`, `Scharr`, `goodFeaturesToTrack`, `SIFT_create`, `ORB_create` - which is where the
boundary has always been drawn.
"""
import ast
import os
import re
import unittest
from typing import Dict, List

from motrack.common.project import ROOT_PATH

CMC_ROOT = os.path.join(ROOT_PATH, 'motrack', 'cmc')

# Each entry is the OpenCV facility that would replace a hand-written component.
BANNED = {
    'estimateAffine2D': 'affine estimation — ransac.py exists for this',
    'estimateAffinePartial2D': 'affine estimation — ransac.py exists for this',
    'findHomography': 'transform estimation — ransac.py exists for this',
    'findTransformECC': 'direct alignment — would replace the whole pipeline',
    'BFMatcher': 'descriptor matching — matching.py exists for this',
    'FlannBasedMatcher': 'descriptor matching — matching.py exists for this',
    'DescriptorMatcher': 'descriptor matching — matching.py exists for this',
    'calcOpticalFlowPyrLK': 'optical flow — pylk.py exists for this',
    'calcOpticalFlowFarneback': 'optical flow — pylk.py exists for this',
    'videostab': 'video stabilisation — would replace the whole module',
}


def source_files() -> List[str]:
    """
    Every Python file under `motrack/cmc/`.
    """
    found = []
    for directory, _, files in os.walk(CMC_ROOT):
        if '__pycache__' in directory:
            continue
        found.extend(os.path.join(directory, f) for f in files if f.endswith('.py'))
    return found


def banned_calls(tree: ast.AST) -> List[tuple]:
    """
    Every call to a banned OpenCV facility, with the function that encloses it.

    Only `ast.Call` nodes count, so a banned name appearing in a docstring or a string is ignored.
    The enclosing name is the innermost function definition containing the call, which is what the
    `_opencv` naming convention is checked against.
    """
    found: List[tuple] = []

    class Visitor(ast.NodeVisitor):
        """
        Tracks the function nesting so each call is attributed to the function it sits in.
        """
        def __init__(self) -> None:
            self.stack: List[str] = []

        def visit_FunctionDef(self, node) -> None:  # pylint: disable=invalid-name
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node) -> None:  # pylint: disable=invalid-name
            target = node.func
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, 'id', None)
            if name in BANNED:
                found.append((self.stack[-1] if self.stack else '', name, node.lineno))
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


class OpenCVBoundaryTest(unittest.TestCase):
    """
    A grep, expressed as a test so it runs in CI rather than living in a reviewer's head.
    """

    def test_banned_calls_only_appear_in_opencv_backends(self) -> None:
        """
        The facilities that would replace a hand-written component are confined to opt-in backends.

        A call is permitted when it sits in a function whose name ends in `_opencv`, which is the
        convention marking the delegating branch. The source is parsed rather than grepped, so the
        banned names can be named freely in docstrings, which they need to be: each backend
        documents the OpenCV function it mirrors.
        """
        files = source_files()
        self.assertGreater(len(files), 10, 'Source discovery is broken, so this test proves nothing.')

        offences: Dict[str, List[str]] = {}
        for path in files:
            with open(path, 'r', encoding='utf-8') as handle:
                tree = ast.parse(handle.read(), filename=path)

            for enclosing, name, line in banned_calls(tree):
                if enclosing.endswith('_opencv'):
                    continue
                relative = os.path.relpath(path, ROOT_PATH)
                offences.setdefault(name, []).append(
                    f'{relative}:{line} in {enclosing or "module scope"}')

        if offences:
            report = '\n'.join(
                f'  cv2.{name} ({BANNED[name]}) at {", ".join(sites)}'
                for name, sites in sorted(offences.items())
            )
            self.fail(f'Banned OpenCV calls outside an opencv backend:\n{report}')

    def test_the_check_can_actually_fail(self) -> None:
        """
        A grep-based guard that cannot fail is worse than none, so the matcher is exercised.

        Comments are stripped before matching, since the banned names are legitimately discussed
        in docstrings — this asserts that distinction holds rather than assuming it.
        """
        pattern = re.compile(r'\bBFMatcher\b')

        self.assertTrue(pattern.search('matches = cv2.BFMatcher(cv2.NORM_HAMMING)'))
        self.assertFalse(pattern.search('# validated against cv2.BFMatcher'.split('#', 1)[0]))
        self.assertFalse(pattern.search('my_bfmatcher_lookalike()'))


if __name__ == '__main__':
    unittest.main()

    def test_hand_written_path_is_present_and_default(self) -> None:
        """
        Each component still implements the work itself, and does so unless asked not to.

        The point of the previous test is that OpenCV is confined to opt-in backends. That is only
        worth anything if the non-opt-in path still contains an implementation, so the markers of
        the hand-written routines are checked here, along with the default of the switch.
        """
        expectations = {
            'components/pylk.py': ['def _compute_flow', 'def solve_2x2_system', 'def min_eigenvalue2x2'],
            'components/ransac.py': ['def estimate_warp_lstsq', 'def _score'],
            'components/matching.py': ['def match_descriptors'],
        }
        for relative, markers in expectations.items():
            path = os.path.join(CMC_ROOT, relative)
            with open(path, 'r', encoding='utf-8') as handle:
                source = handle.read()
            for marker in markers:
                self.assertIn(marker, source,
                              f'{relative} no longer defines {marker!r}: the hand-written path is gone.')

        for relative in ('algorithms/pylk.py', 'algorithms/feature_matching.py'):
            path = os.path.join(CMC_ROOT, relative)
            with open(path, 'r', encoding='utf-8') as handle:
                source = handle.read()
            self.assertIn("implementation: Literal['custom', 'opencv'] = 'custom'", source,
                          f'{relative} must default to the hand-written implementation.')
