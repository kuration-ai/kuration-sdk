"""Output presets: convenient constructors for ordinary :class:`OutputSpec` objects."""
from __future__ import annotations

from .spec import OutputSpec
from ..registry import default_registry, register_preset, validate_parameters


def _modalities(schema, *, categorical: str, categorical_encoders=None,
                categorical_vocabs=None) -> tuple[dict, list[str]]:
    if categorical not in {"error", "omit", "integer-vocab", "encoder"}:
        raise ValueError("categorical must be error, omit, integer-vocab, or encoder")
    categorical_encoders = categorical_encoders or {}
    categorical_vocabs = categorical_vocabs or {}
    mods, omitted = {}, []
    for name, fs in {**schema.fields, **schema.globals}.items():
        if fs.kind == "categorical":
            if categorical == "error":
                raise ValueError(f"output preset encountered categorical field {name!r}; "
                                 "choose categorical='omit' or provide an encoder")
            if categorical == "omit":
                omitted.append(name)
                continue
            if categorical == "integer-vocab":
                if name not in categorical_vocabs:
                    raise ValueError(f"integer-vocab needs categorical_vocabs[{name!r}]")
                mods[name] = {"dtype": "int64", "shape": [], "source": f"vocab:{name}"}
                continue
            hook = categorical_encoders.get(name)
            if not hook:
                raise ValueError(f"categorical='encoder' needs categorical_encoders[{name!r}]")
            mods[name] = {"dtype": "int64", "shape": [], "source": f"encoder:{hook}"}
            continue
        mods[name] = {"dtype": "float32", "shape": [fs.width] if fs.kind == "vector" else [],
                      "source": f"field:{name}"}
    return mods, omitted


def _build(schema, *, container, shard_length, short_policy, categorical=None,
           categorical_encoders=None, categorical_vocabs=None, presence_mask=True, **_kw):
    categorical_fields = [f for f in schema.fields.values() if f.kind == "categorical"]
    if categorical is None:
        if categorical_fields:
            raise ValueError("output preset requires an explicit categorical policy")
        categorical = "omit"
    mods, _ = _modalities(schema, categorical=categorical,
                          categorical_encoders=categorical_encoders,
                          categorical_vocabs=categorical_vocabs)
    if not mods:
        raise ValueError("output preset resolved to no modalities")
    registry = _kw.pop("_registry", None)
    return OutputSpec.from_dict({"modalities": mods, "container": container,
                                 "shard_length": shard_length, "short_policy": short_policy,
                                 "presence_mask": presence_mask,
                                 "vocabs": categorical_vocabs or {}}, registry=registry)


def _numpy(schema, _registry=None, **params):
    return _build(schema, container=params.pop("container", "npz"),
                  shard_length=params.pop("shard_length", 64),
                  short_policy=params.pop("short_policy", "keep"),
                  _registry=_registry, **params)


def _pytorch(schema, _registry=None, **params):
    return _build(schema, container=params.pop("container", "npz"),
                  shard_length=params.pop("shard_length", 64),
                  short_policy=params.pop("short_policy", "pad"),
                  _registry=_registry, **params)


def _tabular(schema, _registry=None, **params):
    return _build(schema, container=params.pop("container", "parquet"),
                  shard_length=params.pop("shard_length", 1024),
                  short_policy=params.pop("short_policy", "keep"),
                  _registry=_registry, **params)


_PARAMS = {"type": "object", "properties": {
    "container": {"type": "string"}, "shard_length": {"type": "integer", "minimum": 1},
    "short_policy": {"enum": ["keep", "pad", "drop"]},
    "categorical": {"enum": ["error", "omit", "integer-vocab", "encoder"]},
    "categorical_encoders": {"type": "object"}, "categorical_vocabs": {"type": "object"},
    "presence_mask": {"type": "boolean"}},
    "additionalProperties": False}


def output_preset(name: str, *, schema, registry=default_registry, **params) -> OutputSpec:
    """Resolve a registered output preset into a validated ``OutputSpec``.

    ``params`` are checked against the preset's descriptor. Unknown presets, missing required
    parameters, and unexpected parameters raise ``ValueError``.
    """
    if name not in registry.names("preset") and registry is default_registry:
        from ..plugins import load_plugins
        load_plugins()
    descriptor = registry.descriptor("preset", name)
    validate_parameters(descriptor, params)
    factory = registry.get("preset", name)
    import inspect
    signature = inspect.signature(factory)
    accepts_registry = "_registry" in signature.parameters
    resolved = factory(schema, **params,
                       **({"_registry": registry} if accepts_registry else {}))
    if isinstance(resolved, dict):
        return OutputSpec.from_dict(resolved, registry=registry)
    if not isinstance(resolved, OutputSpec):
        raise ValueError(f"output preset {name!r} must return OutputSpec or dict, got "
                         f"{type(resolved).__name__}")
    # Reparse direct dataclass construction so it cannot bypass dtype/name/container checks.
    return OutputSpec.from_dict(resolved.to_dict(), registry=registry)


register_preset("numpy-timeseries", _numpy, description="NumPy NPZ time-series shards",
                parameters=_PARAMS, capabilities={"streaming": True}, version="1")
register_preset("pytorch-sequences", _pytorch, description="Padded sequence shards for PyTorch",
                parameters=_PARAMS, capabilities={"streaming": True}, version="1")
register_preset("tabular", _tabular, description="Flat Parquet time-series shards",
                parameters=_PARAMS, capabilities={"streaming": True}, extra="parquet",
                install_hint="pip install 'kuration-sdk[parquet]'", version="1")
