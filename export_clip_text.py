import torch
import torch.nn as nn
import open_clip
import onnx

class TextEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.attn_mask = model.attn_mask
        
    def forward(self, text_tokens, eot_indices):
        x = self.token_embedding(text_tokens).to(torch.float32)
        x = x + self.positional_embedding.to(torch.float32)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eot_indices]  # Use provided indices
        x = x @ self.text_projection
        return x

# 1. Load the OpenCLIP model and tokenizer
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
model = model.to('cpu')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 2. Create the wrapper model
text_encoder_wrapper = TextEncoderWrapper(model)
text_encoder_wrapper.eval()

# 3. Create dummy input
dummy_text_tokens = tokenizer(["a photo of a cat"])
print(f"✅ Created dummy input tensor with shape: {dummy_text_tokens.shape}")

EOS_ID = 49407
dummy_eot_indices = (dummy_text_tokens == EOS_ID).long().argmax(dim=-1)

# 4. Test the wrapper to make sure it works
with torch.no_grad():
    # Test wrapper output
    wrapper_output = text_encoder_wrapper(dummy_text_tokens, dummy_eot_indices)
    
    # Test original model output for comparison
    original_output = model.encode_text(dummy_text_tokens)
    
    # Check if outputs are close (they should be identical)
    if torch.allclose(wrapper_output, original_output, atol=1e-6):
        print("✅ Wrapper model output matches original model output")
    else:
        print("❌ Warning: Wrapper output differs from original")
        print(f"Max difference: {torch.max(torch.abs(wrapper_output - original_output))}")

# 5. Export to ONNX
output_path = "text_encoder.onnx"


torch.onnx.export(
    text_encoder_wrapper,
    (dummy_text_tokens, dummy_eot_indices),
    output_path,
    export_params=True,
    opset_version=14,  # Use opset 11 for better compatibility
    do_constant_folding=True,
    input_names=['text_tokens', 'eot_indices'],
    output_names=['text_features'],
    dynamic_axes={
        'text_tokens': {0: 'batch_size'},
        'eot_indices': {0: 'batch_size'},
        'text_features': {0: 'batch_size'}
    }
)
print(f"✅ Successfully exported model to {output_path}")

# 6. Verify the ONNX model
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model validation passed")
    

import onnxruntime as ort

# Create inference session
ort_session = ort.InferenceSession(output_path)

# Prepare inputs
ort_inputs = {
    'text_tokens': dummy_text_tokens.numpy(),
    'eot_indices': dummy_eot_indices.numpy()
}

# Run inference
ort_outputs = ort_session.run(None, ort_inputs)
print(f"Output shape: {ort_outputs[0].shape}")

print(ort_outputs)

