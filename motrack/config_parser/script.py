"""
Utility scripts config.
"""
from dataclasses import dataclass, field


@dataclass
class DetectionYoloScriptConfig:
    use_symlink: bool = field(default=True)
    skip_empty_images: bool = field(default=False)


@dataclass
class DetectionScriptConfig:
    yolo: DetectionYoloScriptConfig = field(default_factory=DetectionYoloScriptConfig)

@dataclass
class ScriptConfig:
    detection: DetectionScriptConfig = field(default_factory=DetectionScriptConfig)