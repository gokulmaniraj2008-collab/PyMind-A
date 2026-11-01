"""Validate portable JSON scene specifications before Blender execution."""

from dataclasses import dataclass


SUPPORTED_OBJECTS = {"cube", "sphere", "cylinder"}


@dataclass(frozen=True)
class SceneObject:
    kind: str
    name: str
    location: tuple[float, float, float]
    scale: tuple[float, float, float]
    color: tuple[float, float, float, float]


def _vector(value, length, field, minimum=None, maximum=None):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} numbers")
    if any(not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{field} values must be numeric")
    result = tuple(float(v) for v in value)
    if minimum is not None and any(v < minimum for v in result):
        raise ValueError(f"{field} values must be >= {minimum}")
    if maximum is not None and any(v > maximum for v in result):
        raise ValueError(f"{field} values must be <= {maximum}")
    return result


def validate_scene_spec(spec):
    """Return normalized scene data or raise ValueError for invalid input."""
    if not isinstance(spec, dict):
        raise ValueError("scene specification must be a JSON object")
    objects = spec.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("scene must contain a non-empty objects list")

    normalized = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"objects[{index}] must be an object")
        kind = item.get("type", "cube")
        if kind not in SUPPORTED_OBJECTS:
            raise ValueError(
                f"objects[{index}].type must be one of {sorted(SUPPORTED_OBJECTS)}"
            )
        name = item.get("name", f"{kind}_{index}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"objects[{index}].name must be a non-empty string")
        normalized.append(
            SceneObject(
                kind=kind,
                name=name,
                location=_vector(item.get("location", [0, 0, 0]), 3,
                                 f"objects[{index}].location"),
                scale=_vector(item.get("scale", [1, 1, 1]), 3,
                              f"objects[{index}].scale", minimum=0.001),
                color=_vector(item.get("color", [0.35, 0.55, 0.85, 1]), 4,
                              f"objects[{index}].color", minimum=0, maximum=1),
            )
        )
    return normalized
