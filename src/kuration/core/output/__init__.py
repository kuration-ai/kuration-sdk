"""The output layer: selected curation -> training shards (declared modalities, pluggable
encoders/containers, hours-budget track fill)."""
from .base import OutputEncoder, PassthroughEncoder, get_encoder, register_encoder
from .dataset import write_dataset
from .fill import fill_to_hours
from .external_fill import external_fill_to_hours as external_fill_to_hours
from .shardwriter import write_track
from .spec import Modality, OutputSpec
from .track import Track, tracks_from_ticks
from .presets import output_preset
from .containers import Container, register_container

__all__ = [
    "OutputSpec", "Modality",
    "OutputEncoder", "PassthroughEncoder", "register_encoder", "get_encoder",
    "Track", "tracks_from_ticks", "write_track",
    "fill_to_hours",
    "write_dataset",
    "output_preset",
    "Container", "register_container",
]
