import torch
import coremltools as ct

# Create the model and load the weights
torch.backends.quantized.engine = 'qnnpack'

# Create the model and load the weights
scripted_model = torch.jit.load("origin_model.pth")

# Create the input image type
input_face = ct.TensorType(name="face_sequences", shape=(5, 6, 96, 96))
input_voice = ct.TensorType(name="audio_sequences", shape=(5, 1, 80, 16))


# Convert the model
coreml_model = ct.convert(
    scripted_model,
    convert_to='mlprogram',
    inputs=[input_voice,input_face],
    minimum_deployment_target=ct.target.iOS17
)

spec = coreml_model.get_spec()


# Save the CoreML model
coreml_model.save('models/my_network_image_origin.mlpackage')
