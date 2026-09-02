# tests/test_readers.py
import glob
import numpy as np
import pytest
from curation.ingest import readers

RAW = "/home/li/datasets/MMFi_dataset/data/MMFi_Dataset/E01/S01/A01"


@pytest.mark.skipif(len(glob.glob(f"{RAW}/wifi-csi/*.mat")) == 0, reason="MMFi data not present")
def test_read_wifi():
    f = sorted(glob.glob(f"{RAW}/wifi-csi/*.mat"))[0]
    x = readers.read_wifi_frame(f)
    assert x.shape == (3, 114, 10)
    assert np.isfinite(x).all()
    assert x.dtype == np.float32


@pytest.mark.skipif(len(glob.glob(f"{RAW}/lidar/*.bin")) == 0, reason="MMFi data not present")
def test_read_lidar():
    f = sorted(glob.glob(f"{RAW}/lidar/*.bin"))[0]
    x = readers.read_lidar_frame(f)
    assert x.shape[1] == 3 and x.shape[0] <= 1536


@pytest.mark.skipif(len(glob.glob(f"{RAW}/mmwave/*.bin")) == 0, reason="MMFi data not present")
def test_read_mmwave():
    f = sorted(glob.glob(f"{RAW}/mmwave/*.bin"))[0]
    x = readers.read_mmwave_frame(f)
    assert x.shape[1] == 5 and x.shape[0] <= 64


@pytest.mark.skipif(len(glob.glob(f"{RAW}/depth/*.png")) == 0, reason="MMFi data not present")
def test_read_depth():
    f = sorted(glob.glob(f"{RAW}/depth/*.png"))[0]
    x = readers.read_depth_frame(f)
    assert x.shape == (1, 224, 224)
    assert x.dtype == np.float32
    assert x.max() <= 100.0  # meters; loose sanity (Kinect outlier pixels up to ~30m seen)
