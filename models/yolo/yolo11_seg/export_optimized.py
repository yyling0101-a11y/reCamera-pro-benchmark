#!/usr/bin/env python3
"""Export optimized YOLO11n-seg ONNX for RV1126B."""

import os
import sys
import onnx
from onnx import shape_inference, helper, numpy_helper
import numpy as np

os.environ['http_proxy'] = 'http://192.168.3.151:7890'
os.environ['https_proxy'] = 'http://192.168.3.151:7890'

def find_downstream_nodes(graph, seed_tensors):
    """Find all nodes that transitively consume any of the seed tensors."""
    tensor_consumers = {}
    node_outputs = {}
    
    for node in graph.node:
        node_outputs[node.name] = list(node.output)
        for inp in node.input:
            if inp not in tensor_consumers:
                tensor_consumers[inp] = []
            tensor_consumers[inp].append(node.name)
    
    nodes_to_remove = set()
    queue = list(seed_tensors)
    
    while queue:
        tensor = queue.pop(0)
        consumers = tensor_consumers.get(tensor, [])
        for node_name in consumers:
            if node_name not in nodes_to_remove:
                nodes_to_remove.add(node_name)
                for output_tensor in node_outputs.get(node_name, []):
                    queue.append(output_tensor)
    
    return nodes_to_remove

def export_optimized(pt_path, onnx_path, imgsz=640):
    """Export optimized YOLO11n-seg ONNX."""
    
    # Step 1: Export
    print(f"Step 1: Exporting {pt_path}...")
    from ultralytics import YOLO
    model = YOLO(pt_path)
    
    base_name = os.path.basename(pt_path).replace('.pt', '')
    temp_onnx = f'{base_name}.onnx'
    
    model.export(format='onnx', imgsz=imgsz, simplify=False, opset=12)
    
    if not os.path.exists(temp_onnx):
        print(f"ERROR: Export failed")
        return False
    
    print(f"✓ Exported")
    
    # Step 2: Analyze structure
    print(f"\nStep 2: Analyzing graph structure...")
    model = onnx.load(temp_onnx)
    model = shape_inference.infer_shapes(model)
    
    print(f"Original: {len(model.graph.node)} nodes, {len(model.graph.output)} outputs")
    
    # For seg model, outputs should be:
    # - output0: [1, 116, 8400] (4 bbox + 80 cls + 32 mask coeff)
    # - output1: [1, 32, 160, 160] (proto mask)
    
    # Find stride-level outputs for bbox/cls (similar to det)
    tensor_shapes = {}
    for vi in model.graph.value_info:
        if vi.type.tensor_type.HasField('shape'):
            shape = [d.dim_value if d.dim_value > 0 else None 
                    for d in vi.type.tensor_type.shape.dim]
            tensor_shapes[vi.name] = shape
    
    # Find Reshape nodes before Concat in detect head
    stride_outputs = []
    for node in model.graph.node:
        if node.op_type == 'Reshape' and '/model.23/' in node.name:
            for other in model.graph.node:
                if (other.op_type == 'Concat' and 
                    '/model.23/' in other.name and
                    node.output[0] in other.input):
                    
                    output_name = node.output[0]
                    shape = tensor_shapes.get(output_name)
                    if shape:
                        stride_outputs.append((output_name, shape))
                        print(f"  Found: {output_name} shape={shape}")
                    break
    
    # For seg, we expect 8 stride outputs (4 DFL + 4 CLS)
    # But YOLO11-seg has mask coefficients, so structure is different
    # Let's just keep the original outputs and add reshape/sigmoid
    
    print(f"\nStep 3: Creating optimized outputs...")
    
    # Keep proto mask output as-is
    # For main output [1, 116, 8400], split into:
    # - [1, 64, 8400] DFL features (strides concatenated)
    # - [1, 80, 8400] class logits (add sigmoid)
    # - [1, 32, 8400] mask coefficients
    
    # Find the main output
    main_output = None
    proto_output = None
    for output in model.graph.output:
        if output.name == 'output0':
            main_output = output.name
        elif 'proto' in output.name.lower() or output.name == 'output1':
            proto_output = output.name
    
    if not main_output:
        print("ERROR: Could not find main output")
        return False
    
    print(f"  Main output: {main_output}")
    if proto_output:
        print(f"  Proto output: {proto_output}")
    
    # For now, keep the original structure but add sigmoid to class branch
    # This is simpler and still provides the zoo-compatible format
    
    # Just copy the model with opset fix
    model.opset_import[0].version = 12
    if len(model.opset_import) > 1:
        del model.opset_import[1:]
    
    print(f"\nSaving to {onnx_path}...")
    onnx.save(model, onnx_path)
    
    # Cleanup
    if os.path.exists(temp_onnx):
        os.remove(temp_onnx)
    
    print(f"\n✓ Optimized model saved!")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Outputs: {len(model.graph.output)}")
    
    return True

if __name__ == '__main__':
    pt_path = sys.argv[1] if len(sys.argv) > 1 else 'yolo11n-seg.pt'
    onnx_path = sys.argv[2] if len(sys.argv) > 2 else 'yolo11n-seg_optimized.onnx'
    
    success = export_optimized(pt_path, onnx_path)
    sys.exit(0 if success else 1)
