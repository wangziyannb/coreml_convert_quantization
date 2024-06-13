import torch
import coremltools as ct

# Create the model and load the weights
traced_model_2x = torch.jit.load("input/realesrgan_trace_2x.pth")
traced_model_4x = torch.jit.load("input/realesrgan_trace_4x.pth")

# Create the input image type
input_img = ct.TensorType(name="img", shape=(1, 3, 864, 576))

for m, n in zip([traced_model_2x, traced_model_4x],
                ['output/realesrgan_trace_2x.mlpackage', 'output/realesrgan_trace_4x.mlpackage']):
    # Convert the model
    coreml_model = ct.convert(
        m,
        convert_to='mlprogram',
        inputs=[input_img],
        minimum_deployment_target=ct.target.iOS17
    )
    # Save the CoreML model
    coreml_model.save(n)
