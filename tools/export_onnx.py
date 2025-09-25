import argparse
import os
import pprint

import torch

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from fp16_utils.fp16util import network_to_half

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Export HigherHRNet to ONNX')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--output', help='output ONNX file name', default='higherhrnet.onnx', type=str)
    parser.add_argument('--opset', help='ONNX opset version', default=12, type=int)
    parser.add_argument('--input-height', help='input height', default=512, type=int)
    parser.add_argument('--input-width', help='input width', default=512, type=int)
    parser.add_argument('--simplify', help='simplify ONNX model', action='store_true')
    parser.add_argument('--verbose', help='verbose output', action='store_true')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs[-1]

class ModelWrapperMultiOutput(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapperMultiOutput, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return tuple(outputs)

def export_to_onnx(model, args):
    dummy_input = torch.randn(1, 3, args.input_height, args.input_width)

    print("Testing model forward pass...")
    with torch.no_grad():
        original_outputs = model(dummy_input)

    print(f"Number of outputs: {len(original_outputs)}")
    for i, output in enumerate(original_outputs):
        print(f"Output {i} shape: {output.shape}")

    if len(original_outputs) > 1:
        use_multi_output = True
        wrapped_model = ModelWrapperMultiOutput(model)
        output_names = [f'output{i}' for i in range(len(original_outputs))]
    else:
        use_multi_output = False
        wrapped_model = ModelWrapper(model)
        output_names = ['output']

    print(f'Exporting to ONNX format with opset {args.opset}...')
    print(f'Input shape: {dummy_input.shape}')

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'}},
        verbose=args.verbose
    )

    print(f'ONNX model saved to: {args.output}')

    if args.simplify:
        try:
            import onnx
            import onnxsim

            print('Simplifying ONNX model...')
            model_onnx = onnx.load(args.output)
            model_simp, check = onnxsim.simplify(model_onnx)
            assert check, "Simplified model check failed"

            simplified_path = args.output.replace('.onnx', '_simplified.onnx')
            onnx.save(model_simp, simplified_path)
            print(f'Simplified ONNX model saved to: {simplified_path}')

        except ImportError:
            print("onnxsim not installed, skipping simplification")
        except Exception as e:
            print(f"Simplification failed: {e}")

    try:
        import onnx
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model check passed!")

        print("\nModel Inputs:")
        for input in onnx_model.graph.input:
            print(f"  Name: {input.name}")
            shape = []
            for dim in input.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(dim.dim_param)
            print(f"  Shape: {shape}")

        print("\nModel Outputs:")
        for output in onnx_model.graph.output:
            print(f"  Name: {output.name}")
            shape = []
            for dim in output.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(dim.dim_param)
            print(f"  Shape: {shape}")

    except ImportError:
        print("ONNX not installed, skipping model verification")
    except Exception as e:
        print(f"Model verification failed: {e}")

def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)

    state_dict = torch.load(cfg.TEST.MODEL_FILE, map_location='cpu')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    model_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        model_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

    if missing_keys:
        print(f'Missing keys: {len(missing_keys)}')
        if len(missing_keys) < 10:
            for key in list(missing_keys)[:10]:
                print(f'  {key}')

    if unexpected_keys:
        if len(unexpected_keys) < 10:
            for key in list(unexpected_keys)[:10]:
                print(f'  {key}')

    dump_input = torch.rand((1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        model.load_state_dict(model_state_dict, strict=True)

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    export_to_onnx(model, args)

if __name__ == '__main__':
    main()
