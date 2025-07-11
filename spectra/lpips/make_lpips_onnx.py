import argparse
import lpips
import os
import torch

def export_alex_onnx(opset_version=11):

    model = lpips.LPIPS(net='alex').eval()
    for param in model.parameters():
        param.requires_grad_(False)

    dummy_h, dummy_w = 64, 64
    dummy_A = torch.randn(1, 3, dummy_h, dummy_w, dtype=torch.float32) * 2.0 - 1.0
    dummy_B = torch.randn(1, 3, dummy_h, dummy_w, dtype=torch.float32) * 2.0 - 1.0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "onnx/lpips_alex.onnx")

    torch.onnx.export(
        model,
        (dummy_A, dummy_B),
        output_path,
        input_names=["inA", "inB"],
        output_names=["lpips_out"],
        dynamic_axes={
            "inA": {2: "H", 3: "W"},
            "inB": {2: "H", 3: "W"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )
    print(f"LPIPS model exported to ONNX at: {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select desired LPIPS model")
    
    export_table = {
        "alex" : (export_alex_onnx, 11)
    }

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        choices = export_table.keys(),
        help = "Which model do you wish to export?"
    )

    parser.add_argument(
        "-o", "--opset_version",
        type=int,
        required=False,
        help="Which opset version do you wish to use?"
    )

    args = parser.parse_args()

    if args.model in export_table.keys():
        export_func = export_table[args.model][0]
        opset_version = export_table[args.model][1]
        
        if args.opset_version != None:
            opset_version = args.opset_version
        
        export_func(opset_version)
        
    else:
        print(f"Error:: No export function exists for model: {args.model}")