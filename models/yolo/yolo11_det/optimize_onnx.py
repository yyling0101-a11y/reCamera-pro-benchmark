#!/usr/bin/env python3
"""
Optimize YOLO11 ONNX for RV1126B NPU inference.

Strategy: Export YOLO11 without bbox decode, output stride-level features.
This matches Rockchip zoo model structure which is optimized for RV1126B.
"""

import onnx
from onnx import helper, TensorProto
import os

def optimize_yolo11_det(input_path, output_path):
    """
    Optimize YOLO11 detection model for RV1126B.
    
    Original output: [1, 84, 8400] (pre-decoded bbox + classes)
    Optimized outputs: stride-level DFL features + class sigmoid + confidence
    """
    print(f"Loading {input_path}...")
    model = onnx.load(input_path)
    graph = model.graph
    
    print(f"Original: {len(graph.node)} nodes, {len(graph.output)} outputs")
    print(f"Original output: {[o.name for o in graph.output]}")
    
    # Find stride-level feature outputs before concat
    # The detect head has:
    # - model.23/Concat: concatenates DFL from 3 strides
    # - model.23/Concat_1: concatenates classes from 3 strides
    
    # We want to cut before these concats and expose stride-level outputs
    
    # Find the concat nodes
    dfl_concat = None
    cls_concat = None
    
    for node in graph.node:
        if node.name == '/model.23/Concat':
            dfl_concat = node
            print(f"Found DFL concat with {len(node.input)} inputs")
        elif node.name == '/model.23/Concat_1':
            cls_concat = node
            print(f"Found CLS concat with {len(node.input)} inputs")
    
    if not dfl_concat or not cls_concat:
        print("ERROR: Could not find required concat nodes")
        return False
    
    # Get stride-level inputs
    dfl_inputs = list(dfl_concat.input)
    cls_inputs = list(cls_concat.input)
    
    print(f"DFL inputs: {dfl_inputs}")
    print(f"CLS inputs: {cls_inputs}")
    
    # Build new outputs
    new_outputs = []
    new_nodes = []
    
    # Add DFL features as outputs (raw features, decode on CPU)
    for i, inp in enumerate(dfl_inputs):
        stride = [8, 16, 32][i]
        output_name = f'dfl_stride_{stride}'
        new_outputs.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None))
    
    # Add Sigmoid to class inputs
    for i, inp in enumerate(cls_inputs):
        stride = [8, 16, 32][i]
        
        # Sigmoid output
        sigmoid_out = f'cls_stride_{stride}_sigmoid'
        sigmoid_node = helper.make_node(
            'Sigmoid',
            inputs=[inp],
            outputs=[sigmoid_out],
            name=f'sigmoid_stride_{stride}'
        )
        new_nodes.append(sigmoid_node)
        new_outputs.append(helper.make_tensor_value_info(sigmoid_out, TensorProto.FLOAT, None))
        
        # ReduceSum for confidence (sum over class dimension)
        reduce_out = f'conf_stride_{stride}'
        reduce_node = helper.make_node(
            'ReduceSum',
            inputs=[sigmoid_out],
            outputs=[reduce_out],
            name=f'reducesum_stride_{stride}',
            axes=[1],
            keepdims=1
        )
        new_nodes.append(reduce_node)
        new_outputs.append(helper.make_tensor_value_info(reduce_out, TensorProto.FLOAT, None))
    
    # Remove decode nodes (Softmax, Slice, Sub, Add, Div after Concat)
    # Keep nodes up to and including the concat inputs
    
    # Find all nodes that are part of decode
    decode_ops = set()
    for node in graph.node:
        if node.op_type in ['Softmax', 'Slice', 'Sub', 'Div']:
            # Check if this is after the concats (part of decode)
            # Simple heuristic: if it's in model.23 and uses concat outputs
            if '/model.23/' in node.name:
                for inp in node.input:
                    if 'Concat' in inp:
                        decode_ops.add(node.name)
                        break
    
    print(f"Identified {len(decode_ops)} decode operations to remove")
    
    # Build new graph
    kept_nodes = [n for n in graph.node if n.name not in decode_ops]
    all_nodes = kept_nodes + new_nodes
    
    new_graph = helper.make_graph(
        all_nodes,
        graph.name,
        graph.input,
        new_outputs,
        graph.initializer
    )
    
    new_model = helper.make_model(new_graph, ir_version=model.ir_version)
    new_model.opset_import[0].version = model.opset_import[0].version
    
    print("Validating...")
    onnx.checker.check_model(new_model)
    
    print(f"Saving to {output_path}...")
    onnx.save(new_model, output_path)
    
    print(f"✓ Optimized: {len(all_nodes)} nodes, {len(new_outputs)} outputs")
    return True

if __name__ == '__main__':
    input_path = '/home/seeed/recamera_pro/benchmark/models/yolo/yolo11_det/yolo11n.onnx'
    output_path = '/home/seeed/recamera_pro/benchmark/models/yolo/yolo11_det/yolo11n_optimized.onnx'
    
    success = optimize_yolo11_det(input_path, output_path)
    if success:
        print("\n✓ Optimization successful!")
    else:
        print("\n✗ Optimization failed!")
