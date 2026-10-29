import numpy as np
import pytest

from lightwork.measure.colour import cell_stats


def test_flat_field_crop_preserves_camera_coordinate_readings():
    image = np.full((40, 50, 3), 170, np.uint8)
    flat = np.full((50, 70, 3), 210, np.uint8)
    flat[:, 25:35] = 190
    local = cell_stats(image, {'A1': (20, 20)}, 5, 'r', flat_field=flat,
                       flat_field_origin=(-10, -5))
    camera = cell_stats(image, {'A1': (620, 100)}, 5, 'r', origin=(600, 80),
                        flat_field=flat, flat_field_origin=(590, 75))
    assert camera == local
    assert camera['A1'].n > 0
    assert camera['A1'].i_lin > cell_stats(image, {'A1': (20, 20)}, 5, 'r')['A1'].i_lin


def test_flat_field_crop_must_cover_keyframe():
    image = np.full((40, 50, 3), 170, np.uint8)
    with pytest.raises(ValueError, match='does not cover'):
        cell_stats(image, {'A1': (620, 100)}, 5, 'r', origin=(600, 80),
                   flat_field=image, flat_field_origin=(601, 80))
