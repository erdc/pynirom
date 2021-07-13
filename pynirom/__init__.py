"""
Tools for non-intrusive reduced order modeling
"""
__all__ = ['pod', 'dmd',
	   ]

#Core imports
from pynirom.pod.pod_utils import (compute_pod_multicomponent,
				   compute_trunc_basis,
				   project_onto_basis,
				   reconstruct_from_rom
				  )

from pynirom.dmd.main import DMDBase 

__version__ = '0.1.0'

### OLD FILE
#from . import dmd
#from . import pod
#from . import rbf

