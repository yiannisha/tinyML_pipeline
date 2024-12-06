import os
import argparse

import torch
from onnxruntime.quantization import quant_pre_process, quantize_dynamic

from utils.core import rootdir, get_checkpoint, validate_exporting_args, pytorch2onnx 
from _types import ExportingArgs, Quantization

def main(args: ExportingArgs):
    
    # find model checkpoint file (pytorch or onnx based on quantization type)
    file_type = 'pkl' if args.quant == Quantization.TORCH_STATIC.value or args.quant == Quantization.TORCH_DYNAMIC.value else 'onnx'
    
    # try to get file from the checkpoint folder
    checkpoint_file = get_checkpoint(args.model, args.checkpoint_id, file_type) 

    if not checkpoint_file: raise ValueError('no latest checkpoint found')
   
    # load checkpoint model
    checkpoint_size = os.path.getsize(checkpoint_file)/1e6 # model size in MB

    if file_type == 'pkl':
        checkpoint_model = torch.load(checkpoint_file)
    # apply quantization
    match args.quant:
        case Quantization.TORCH_STATIC.value:
            # use "qnnpack" for arm architectures and "x86" for x86 architectures
            backend = "qnnpack"
            checkpoint_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
            torch.backends.quantized.engine = backend
            quantized_model = torch.quantization.prepare(checkpoint_model, inplace=False)
            quantized_model = torch.quantization.convert(quantized_model, inplace=False)
        case Quantization.TORCH_DYNAMIC.value:
            quantized_model = torch.quantization.quantize_dynamic(
                checkpoint_model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8,
            )
        case Quantization.ONNX_STATIC.value:
            # @TODO: support static quantization for onnx
            raise NotImplemented("static quantization for onnx not supported yet")
        case Quantization.ONNX_DYNAMIC.value:
            # this is handled later
            pass

    # convert back to onnx if necessary
    new_name = f'{args.model}_{args.checkpoint_id}_{args.quant}'
    onnx_out_path = os.path.join(rootdir, 'checkpoints', args.model, f'{new_name}.onnx')
    out_file_size = None
    if file_type == "pkl":
        # save quantized model both as pkl and onnx
        torch.save(quantized_model, os.path.join(rootdir, 'checkpoints', args.model, f'{new_name}.pkl'))
        out_file_size = os.path.getsize(os.path.join(rootdir, 'checkpoints', args.model, f'{new_name}.pkl'))/1e6
        # @TODO: find a way to at least try and export quantized models to onnx
        # @TODO: make shape generalizable
        # pytorch2onnx(quantized_model, onnx_out_path, (1250, 1))
    else:
        # save quantized model only as onnx
        quant_pre_process(checkpoint_file, 'temp.onnx')
        # apply dynamic quantization
        quantized_model = quantize_dynamic('temp.onnx', onnx_out_path)
        os.remove('temp.onnx')

        out_file_size = os.path.getsize(onnx_out_path)/1e6

    print(f'Input file size: {checkpoint_size:.2f} MB')
    print(f'Output file size: {out_file_size:.2f} MB')

    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, help='model name')
    argparser.add_argument('--checkpoint-id', type=str, help='checkpoint id', default='latest')
    argparser.add_argument('--quant', type=str, help='quantization type. Possible values: torch-static, torch-dynamic, onnx-static, onnx-dynamic', default='pytorch')

    args = ExportingArgs(**vars(argparser.parse_args()))

    validate_exporting_args(args)

    main(args)
