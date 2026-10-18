import coremltools as ct
import os
try:
    out = ct.models.MLModel.compileModel('DepthAnythingV2SmallF16.mlpackage')
    print("COMPILED TO:", out)
except Exception as e:
    print("ERROR:", e)
