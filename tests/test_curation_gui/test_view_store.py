"""Tests for curation.gui.core.view_store — persistent 3D camera views."""
from __future__ import annotations

import os

from curation.gui.core.view_store import (
    clear_view,
    load_view,
    save_view,
)

CAM = {"eye": {"x": 2.5, "y": 1.0, "z": 2.0},
       "up": {"x": 0.0, "y": 0.0, "z": 1.0},
       "center": {"x": 0.0, "y": 0.0, "z": 0.0}}


def test_save_load_roundtrip(tmp_path):
    save_view("mmfi_v4", "E04_S33_A01", "lidar", CAM, views_dir=str(tmp_path))
    got = load_view("mmfi_v4", "E04_S33_A01", "lidar", views_dir=str(tmp_path))
    assert got == CAM


def test_load_missing_returns_none(tmp_path):
    assert load_view("mmfi_v4", "nope", "lidar", views_dir=str(tmp_path)) is None


def test_save_two_modalities_independent(tmp_path):
    save_view("mmfi_v4", "E01", "lidar", CAM, views_dir=str(tmp_path))
    save_view("mmfi_v4", "E01", "mmwave",
              {"eye": {"x": 1.0, "y": 1.0, "z": 1.0}},
              views_dir=str(tmp_path))
    assert load_view("mmfi_v4", "E01", "lidar", views_dir=str(tmp_path)) == CAM
    assert load_view("mmfi_v4", "E01", "mmwave",
                     views_dir=str(tmp_path)) == {"eye": {"x": 1.0, "y": 1.0, "z": 1.0}}


def test_clear_view(tmp_path):
    save_view("mmfi_v4", "E01", "lidar", CAM, views_dir=str(tmp_path))
    clear_view("mmfi_v4", "E01", "lidar", views_dir=str(tmp_path))
    assert load_view("mmfi_v4", "E01", "lidar", views_dir=str(tmp_path)) is None


def test_persists_across_save(tmp_path):
    # saving overwrites prior value for the same (sample, modality)
    save_view("mmfi_v4", "E01", "lidar", CAM, views_dir=str(tmp_path))
    save_view("mmfi_v4", "E01", "lidar",
              {"eye": {"x": 9.9, "y": 0, "z": 0}},
              views_dir=str(tmp_path))
    assert load_view("mmfi_v4", "E01", "lidar",
                     views_dir=str(tmp_path)) == {"eye": {"x": 9.9, "y": 0, "z": 0}}
