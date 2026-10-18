import coremltools as ct
from PIL import Image
import numpy as np
import matplotlib

def colorize(depth):
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    colored = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)

print("Loading Core ML model...")
model = ct.models.MLModel('DepthAnythingV2SmallF16.mlpackage')
print("Model loaded successfully.")

spec = model.get_spec()
print("Input details:")
for inp in spec.description.input:
    if inp.type.WhichOneof('Type') == 'imageType':
        print(f"Image [{inp.name}] size: {inp.type.imageType.width}x{inp.type.imageType.height}")
    else:
        print(f"[{inp.name}] type: {inp.type}")

img_path = 'assets/examples/demo01.jpg'
print(f"Loading {img_path}")
img = Image.open(img_path).convert('RGB')
orig_w, orig_h = img.size

# Assume 518x518 default but try to fetch from spec
w, h = 518, 518
try:
    if len(spec.description.input) > 0 and spec.description.input[0].type.WhichOneof('Type') == 'imageType':
        w_expected = spec.description.input[0].type.imageType.width
        h_expected = spec.description.input[0].type.imageType.height
        if w_expected > 0 and h_expected > 0:
            w, h = w_expected, h_expected
except:
    pass

print(f"Resizing image to {w}x{h}")
img_resized = img.resize((w, h), Image.Resampling.BICUBIC)

print("Running inference...")
out = model.predict({'image': img_resized})

depth = out['depth']
if isinstance(depth, Image.Image):
    depth = np.array(depth)
elif isinstance(depth, np.ndarray):
    # Depending on shape, squeeze
    depth = np.squeeze(depth)

print(f"Output depth shape: {depth.shape}")

depth_resized = Image.fromarray(depth.astype("float32")).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
depth_resized = np.array(depth_resized)

colored_depth = colorize(depth_resized)
out_file = 'coreml_output.png'
colored_depth.save(out_file)
print(f"Saved inference result to {out_file}")
