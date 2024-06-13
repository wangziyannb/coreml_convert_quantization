import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models
import coremltools as ct
model = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights="coco_pose")
model.prep_model_for_conversion(input_size=[1, 3, 640, 640])

traced_model = torch.jit.trace(model, torch.randn(1, 3, 640, 640))

torch.jit.save(traced_model, "yolo_nas_pose_s_nonms.pth")
traced_model=torch.jit.load("yolo_nas_pose_s_nonms.pth")

input_image = ct.ImageType(name="my_input", shape=(1, 3, 640, 640), scale=1/255)

# Convert the model
coreml_model = ct.convert(
    traced_model,
    convert_to='mlprogram',
    inputs=[input_image],
    minimum_deployment_target=ct.target.iOS17
)

coreml_model.save('models/yolo_nas_pose_s_nonms.mlpackage')
