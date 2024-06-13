import torch
import coremltools as ct

# Create the model and load the weights
torch.backends.quantized.engine = 'qnnpack'

# Create the model and load the weights
scripted_model = torch.jit.load("input/origin_wave2lip_model.pth")
scripted_model_quantized = torch.jit.load("input/quantized_wave2lip_model.pth")

# Create the input image type
input_face = ct.TensorType(name="face_sequences", shape=(5, 6, 96, 96))
input_voice = ct.TensorType(name="audio_sequences", shape=(5, 1, 80, 16))

for m, n in zip([scripted_model, scripted_model_quantized],
                ['output/origin_wave2lip_model.mlpackage', 'output/quantized_wave2lip_model.mlpackage']):
    # Convert the model
    coreml_model = ct.convert(
        m,
        convert_to='mlprogram',
        inputs=[input_voice, input_face],
        minimum_deployment_target=ct.target.iOS17
    )
    # Save the CoreML model
    coreml_model.save(n)
