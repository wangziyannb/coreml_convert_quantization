import torch
import coremltools as ct

# Create the model and load the weights
scripted_model = torch.jit.load("realesrgan_trace_2x.pth")

# Create the input image type
input_img = ct.TensorType(name="img", shape=(1, 3, 864, 576))
# input_voice = ct.TensorType(name="audio_sequences", shape=(5, 1, 80, 16))


# Convert the model
coreml_model = ct.convert(
    scripted_model,
    convert_to='mlprogram',
    inputs=[input_img],
    minimum_deployment_target=ct.target.iOS17
)
# Save the CoreML model
coreml_model.save('models/realesrgan_trace_2x.mlpackage')
