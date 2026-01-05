from dataclasses import dataclass, field
from pathlib import Path
import typing as t
import yaml

@dataclass
class BasePaths:
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    model_data_dir = "data/datasets"
    train_data_dir = "data/datasets/train"
    val_data_dir = "data/datasets/val"
    test_data_dir = "data/datasets/test"

@dataclass
class BuildConfig:
    pdc_window_days = 90
    max_sequence_days = 128
    dx_adherence_threshold = 0.8
    phq9_spike = 5.0
    utilization_spike = 2.0
    relapse_horizon_days = 30

@dataclass
class DatasetConfig:
    train_split = 0.75
    val_split = 0.10
    test_split = 0.15

@dataclass
class Config:
    seed: int = field(default=42)
    paths: BasePaths = field(default_factory=BasePaths)
    build: BuildConfig = field(default_factory=BuildConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


def load_config(path: str, model: str = "") -> Config:
    def _apply_yaml(obj, data = {}):
        for key, val in (data or {}).items():
            if val is None or not hasattr(obj, key):
                continue
            current = getattr(obj, key)
            if hasattr(current, "__dataclass_fields__"):
                _apply_yaml(current, val or {})
                continue
            if key.endswith("_dir") or key.endswith("_path"):
                setattr(obj, key, _as_path(val))
            else:
                setattr(obj, key, val)
        return obj
    
    def _as_path(val: t.Optional[t.Union[str, Path]]) -> t.Optional[Path]:
        if val is None or isinstance(val, Path):
            return val
        return Path(val).expanduser()
    
    if not ("." in path and path.endswith(".yaml")):
        cfg_path = f'{path}.yaml'

    cfg_path = Path(cfg_path)
    if not Path(cfg_path).exists():
        raise NameError(f"The config file {cfg_path} does not exist.")
    
    cfg = Config()
    data = yaml.safe_load(cfg_path.read_text()) or {}
    _apply_yaml(cfg, data)

    return cfg


