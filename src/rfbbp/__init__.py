import numpy as np

from rfbbp.rfbbp import RFBBPPyMC
from rfbbp.rfbbpcox import RFBBPCox
from rfbbp.rfbbprcs import RFBBPRCS
from rfbbp.rfbbpquadratic import RFBBPQuadratic
try:
    from rfbbp.rfbbp_vi_irls import RFBBPVIIRLS
except ModuleNotFoundError:
    RFBBPVIIRLS = None
try:
    from rfbbp.rfbbp_vb_newton import RFBBPVBNewton
except ModuleNotFoundError:
    RFBBPVBNewton = None

def main() -> None:
    print("Hello from rfbbp!")
    print(np.random.rand(6))
