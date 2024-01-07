"""
Tracklet Dataclass.
"""
import enum
from multiprocessing import Value, Lock
from typing import Tuple, ClassVar, Optional, List, Any, Union

from motrack.library.cv.bbox import PredBBox

_TRACKLET_DEFAULT_MAX_HISTORY = 32

TrackletHistoryType = List[Tuple[int, PredBBox]]


# Conventions
class TrackletCommonData(enum.Enum):
    """
    Tracklet common data type used.
    """
    APPEARANCE = 'appearance'
    APPEARANCE_BUFFER = 'appearance-buffer'


class TrackletState(enum.Enum):
    """
    Tracklet state:
    - NEW: New detected object (not confirmed)
    - ACTIVE: Active tracklet (confirmed)
    - LOST: Inactive tracklet (lost due occlusion, etc.)
    """
    NEW = enum.auto()
    ACTIVE = enum.auto()
    LOST = enum.auto()
    DELETED = enum.auto()


class Tracklet:
    """
    Tracking object state
    """
    Counter: ClassVar[Value] = Value('i', 0)
    CounterLock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        bbox: PredBBox,
        frame_index: int,
        max_history: int = _TRACKLET_DEFAULT_MAX_HISTORY,
        _id: Optional[int] = None,
        state: TrackletState = TrackletState.NEW
    ):
        """
        Initializes new tracklet.

        Args:
            bbox: Initial bbox
            frame_index: Frame index when tracklet was created
            max_history: Max tracklet history (old data is deleted)
            _id: Set tracklet if explicitly
            state: Initial state

        Returns:
            Initialized tracklet
        """
        if _id is None:
            with Tracklet.CounterLock:
                Tracklet.Counter.value += 1
                self._id = Tracklet.Counter.value
        else:
            self._id = _id

        # Update bbox
        bbox.id = self._id

        self._max_history = max_history
        self._start_frame_index = frame_index

        # State
        self._history: TrackletHistoryType = [(frame_index, bbox)]
        self._total_matches = 1
        self._total_lost = 0
        self._state = state
        self._data = None

    def __repr__(self) -> str:
        return f'Tracklet(id={self._id}, bbox={self.bbox}, state={self._state})'

    def __hash__(self) -> int:
        return self._id

    @property
    def state(self) -> TrackletState:
        """
        Gets tracklet state.

        Returns:
            Tracklets state.
        """
        return self._state

    @state.setter
    def state(self, new_state: TrackletState) -> None:
        """
        Sets tracklet state.

        Args:
            new_state: New tracklet state
        """
        self._state = new_state

    @property
    def is_tracked(self) -> bool:
        """
        Checks if tracklet is being tracked.

        Returns:
            True if tracklets is tracked else False
        """
        return self._state in [TrackletState.ACTIVE, TrackletState.LOST]

    @property
    def start_frame_index(self) -> int:
        """
        Gets frame when the Tracklet was initialized.

        Returns:
            Tracklet first frame.
        """
        return self._start_frame_index

    @property
    def age(self) -> int:
        """
        Gets tracklet age.

        Returns:
            Tracklet age
        """
        return self.frame_index - self._start_frame_index

    @property
    def lost_time(self) -> int:
        """
        Gets number of frames that tracklet has LOST state

        Example: (match, missing, match, missing, missing) -> returns 2

        Returns:
            Number of unmatched frames
        """
        return self._total_lost

    @property
    def id(self) -> int:
        """
        Gets tracklet id.

        Returns:
            Tracklet id
        """
        return self._id


    @property
    def latest(self) -> Tuple[int, PredBBox]:
        """
        The latest detection of one tracklet.

        Returns:
            id and PredBBox
        """
        return self._history[-1]

    @property
    def first(self) -> Tuple[int, PredBBox]:
        """
        First detection (last detection in history) of one tracklet.

        Returns:
            id and PredBBox
        """
        return self._history[0]

    @property
    def history(self) -> TrackletHistoryType:
        """
        Gets full tracklet history.

        Returns:
            Tracklet history
        """
        return self._history

    @property
    def bbox(self) -> PredBBox:
        """
        Gets current tracklet bbox.

        Returns:
            Current bbox
        """
        return self.latest[1]

    @property
    def frame_index(self) -> int:
        """
        Gets current (last) frame index.

        Returns:
            Frame index
        """
        return self.latest[0]

    @property
    def total_matches(self) -> int:
        """
        Total number of tracklet matches with detected object.

        Returns:
            Total number of tracklet matches
        """
        return self._total_matches

    def __len__(self) -> int:
        """
        Returns:
            tracklet length
        """
        return len(self._history)

    def update(self, bbox: PredBBox, frame_index: int, state: Optional[TrackletState] = None) -> 'Tracklet':
        """
        Updates tracklet BBox (and history).
        Disclaimer: Updates bbox id!

        Args:
            bbox: New bbox
            frame_index: Current frame index
            state: New tracklet state
        """
        bbox.id = self._id  # Update bbox id

        self._history.append((frame_index, bbox))

        # Remove `obsolete` history
        while len(self._history) > self._max_history:
            self._history.pop(0)

        if state is not None:
            self._state = state

        # Update state stats
        if self._state in [TrackletState.NEW, TrackletState.ACTIVE]:
            self._total_matches += 1
            self._total_lost = 0
        elif self._state == TrackletState.LOST:
            self._total_lost += 1

        return self

    def set(self, key: Union[str, TrackletCommonData], value: Any) -> None:
        """
        Updates Tracklet key-value data.

        Args:
            key: Data key
            value: Data value
        """
        if self._data is None:
            self._data = {}
        self._data[key] = value

    def get(self, key: Union[str, TrackletCommonData]) -> Any:
        """
        Gets Tracklet data by key.

        Args:
            key: Data key

        Returns:
            Data value if key is in data else None
        """
        if self._data is None or key not in self._data:
            return None

        return self._data[key]
