import os
import torch
import torch.nn.functional as F
import onnx
from torchinfo import summary
from baseline.model.DeepMAR import DeepMAR_ResNet50  # Adjust import based on your project structure

# Subclass for ONNX export with fixed pooling and sigmoid activation
class DeepMAR_ResNet50_Export(DeepMAR_ResNet50):
    def forward(self, x):
        x = self.base(x)
        # Fixed kernel size for 224x224 input (feature map is 7x7 after ResNet-50)
        x = F.avg_pool2d(x, (7, 7))
        x = x.view(x.size(0), -1)
        if self.drop_pool5:
            x = F.dropout(x, p=self.drop_pool5_rate, training=self.training)
        x = self.classifier(x)
        x = torch.sigmoid(x)  # Sigmoid for multi-label confidence scores
        return x

def export_deepmar_to_onnx(model_path, onnx_output_path, num_att):
    """
    Export DeepMAR ResNet-50 model to ONNX format.

    Args:
        model_path (str): Path to PyTorch model checkpoint.
        onnx_output_path (str): Path to save ONNX model.
        num_att (int): Number of attributes (output classes).
    """
    # Instantiate export-friendly model
    model = DeepMAR_ResNet50_Export(num_att=num_att, last_conv_stride=2)
    model = model.cuda()
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get("state_dicts", checkpoint)
    if isinstance(state_dict, list):
        state_dict = state_dict[0]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    # Dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Dummy input test output shape: {output.shape}")
    print(f"Sample output (with sigmoid): {output[0][:5]}")

    # Print model summary
    print("Model Summary:")
    summary(model, input_size=(1, 3, 224, 224))

    # Export to ONNX
    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
    print(f"Exporting to ONNX: {onnx_output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )
    print(f"Model exported to {onnx_output_path}")

    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification completed!")

if __name__ == "__main__":
    model_path = "exp/deepmar_resnet50/peta/partition0/run1/model/ckpt_epoch150.pth"  # Update as needed
    onnx_output_path = "onnx_models/deepmar.onnx"
    num_att = 45  # Adjust based on your training configuration
    export_deepmar_to_onnx(model_path, onnx_output_path, num_att)