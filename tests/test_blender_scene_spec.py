import pytest

from pymind.integrations.blender.scene_spec import validate_scene_spec


def test_valid_scene_is_normalized():
    objects = validate_scene_spec({
        "objects": [{"type": "cube", "name": "Body", "location": [1, 2, 3]}]
    })
    assert len(objects) == 1
    assert objects[0].name == "Body"
    assert objects[0].location == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("spec", [{}, {"objects": []}, {"objects": "cube"}])
def test_rejects_missing_or_invalid_objects(spec):
    with pytest.raises(ValueError):
        validate_scene_spec(spec)


def test_rejects_unsupported_object_type():
    with pytest.raises(ValueError):
        validate_scene_spec({"objects": [{"type": "monkey"}]})


def test_rejects_invalid_color_range():
    with pytest.raises(ValueError):
        validate_scene_spec({"objects": [{"color": [2, 0, 0, 1]}]})
