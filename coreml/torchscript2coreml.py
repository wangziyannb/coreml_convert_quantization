import torch
import coremltools as ct

# Create the model and load the weights
torch.backends.quantized.engine = 'qnnpack'

# Create the model and load the weights
scripted_model = torch.jit.load("quantized_model_wave2lip.pth")

# Create the input image type
input_image = ct.ImageType(name="my_input", shape=(1, 3, 32, 32), scale=1/255)



# Convert the model
coreml_model = ct.convert(
    scripted_model,
    convert_to='mlprogram',
    inputs=[input_image, input_image],
    minimum_deployment_target=ct.target.iOS17
)
# coreml_model = ct.convert(scripted_model,
#               convert_to='neuralnetwork',
#               inputs=[input_image])

# Modify the output's name to "my_output" in the spec
spec = coreml_model.get_spec()
# ct.utils.rename_feature(spec, "dequantize_6", "my_output")
#

# Re-create the model from the updated spec
coreml_model_updated = ct.models.MLModel(spec, weights_dir='quantized_model_wave2lip.pth')

# Save the CoreML model
coreml_model_updated.save('models/my_network_image.mlpackage')

# import torch
# import coremltools as ct
# from model import MyNet
#
# # Create the model and load the weights
# model = MyNet()
# model.load_state_dict(torch.load('quantized_model_wave2lip.pth', map_location=torch.device('cpu')))
# model.eval()
# # Script the model
# scripted_model = torch.jit.trace(model, torch.randn(1, 3, 32, 32))
#
# # Create the input image type
# input_image = ct.ImageType(name="my_input", shape=(1, 3, 32, 32), scale=1/255)
#
# # Convert the model
# coreml_model = ct.convert(scripted_model,
#               convert_to='mlprogram',
#               inputs=[input_image],minimum_deployment_target=ct.target.iOS17)
#
# # Modify the output's name to "my_output" in the spec
# spec = coreml_model.get_spec()
# ct.utils.rename_feature(spec, "linear_2", "my_output")
# #
#
# # Re-create the model from the updated spec
# coreml_model_updated = ct.models.MLModel(spec, weights_dir='models/my_network.pth')
#
# # Save the CoreML model
# coreml_model_updated.save('models/my_network_image.mlpackage')