import os, sys
import pytest

# Tests run from the CondMDI repository root (they import CondMDI modules and read dataset/).
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)

CLIPS = ['000021', '012314']


@pytest.fixture(scope='session', params=CLIPS)
def motion(request):
    from motion_project.hpp.humanml3d_loader import load_humanml3d_motion
    return load_humanml3d_motion(request.param)
