#!/usr/bin/env python3
"""
Export optimized YOLO11 ONNX for RV1126B.

Strategy: Match Rockchip zoo model structure exactly:
- Output stride-level features with spatial dimensions [1, 64, H, W]
- Add Sigmoid to class outputs
- Add ReduceSum to get confidence scores
"""

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
    """Export optimized YOLO11 ONNX matching zoo structure."""
    
    # Step 1: Export with ultralytics
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
    
    # Step 2: ONNX surgery
    print(f"\nStep 2: Optimizing graph structure...")
    model = onnx.load(temp_onnx)
    model = shape_inference.infer_shapes(model)
    
    print(f"Original: {len(model.graph.node)} nodes")
    
    # Find stride-level outputs
    tensor_shapes = {}
    for vi in model.graph.value_info:
        if vi.type.tensor_type.HasField('shape'):
            shape = [d.dim_value if d.dim_value > 0 else None 
                    for d in vi.type.tensor_type.shape.dim]
            tensor_shapes[vi.name] = shape
    
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
    
    if len(stride_outputs) < 6:
        print(f"ERROR: Only found {len(stride_outputs)} stride outputs")
        return False
    
    # Identify stride sizes
    stride_sizes = [(80, 80), (40, 40), (20, 20)]  # for 640x640 input
    
    # Step 3: Remove decode operations
    print(f"\nStep 3: Removing decode operations...")
    seed_tensors = [name for name, _ in stride_outputs]
    nodes_to_remove = find_downstream_nodes(model.graph, seed_tensors)
    print(f"✓ Will remove {len(nodes_to_remove)} nodes")
    
    # Step 4: Add reshape, sigmoid, and reducesum nodes
    print(f"\nStep 4: Adding post-processing nodes...")
    new_nodes = []
    new_outputs = []
    
    for i, (output_name, shape) in enumerate(stride_outputs):
        stride_idx = i % 3  # 0, 1, 2 for stride 8, 16, 32
        is_dfl = i < 3
        
        H, W = stride_sizes[stride_idx]
        
        if is_dfl:
            # DFL features: reshape to [1, 64, H, W]
            # The flattened shape is [1, 64, H*W]
            new_shape = [1, 64, H, W]
            reshaped_name = f'dfl_stride_{stride_idx}_reshaped'
            
            # Create reshape node
            shape_tensor = numpy_helper.from_array(np.array(new_shape, dtype=np.int64), 
                                                   name=f'dfl_stride_{stride_idx}_shape')
            model.graph.initializer.append(shape_tensor)
            
            reshape_node = helper.make_node(
                'Reshape',
                inputs=[output_name, shape_tensor.name],
                outputs=[reshaped_name],
                name=f'reshape_dfl_{stride_idx}'
            )
            new_nodes.append(reshape_node)
            new_outputs.append(helper.make_tensor_value_info(
                reshaped_name, onnx.TensorProto.FLOAT, new_shape
            ))
            print(f"  DFL stride {stride_idx}: reshape to {new_shape}")
        else:
            # Class features: reshape + sigmoid + reducesum
            new_shape = [1, 80, H, W]
            reshaped_name = f'cls_stride_{stride_idx}_reshaped'
            
            # Reshape
            shape_tensor = numpy_helper.from_array(np.array(new_shape, dtype=np.int64),
                                                   name=f'cls_stride_{stride_idx}_shape')
            model.graph.initializer.append(shape_tensor)
            
            reshape_node = helper.make_node(
                'Reshape',
                inputs=[output_name, shape_tensor.name],
                outputs=[reshaped_name],
                name=f'reshape_cls_{stride_idx}'
            )
            new_nodes.append(reshape_node)
            
            # Sigmoid
            sigmoid_name = f'cls_stride_{stride_idx}_sigmoid'
            sigmoid_node = helper.make_node(
                'Sigmoid',
                inputs=[reshaped_name],
                outputs=[sigmoid_name],
                name=f'sigmoid_cls_{stride_idx}'
            )
            new_nodes.append(sigmoid_node)
            new_outputs.append(helper.make_tensor_value_info(
                sigmoid_name, onnx.TensorProto.FLOAT, new_shape
            ))
            
            # ReduceSum for confidence (sum over class dimension)
            conf_name = f'conf_stride_{stride_idx}'
            reduce_node = helper.make_node(
                'ReduceSum',
                inputs=[sigmoid_name],
                outputs=[conf_name],
                name=f'reducesum_conf_{stride_idx}',
                axes=[1],
                keepdims=1
            )
            new_nodes.append(reduce_node)
            new_outputs.append(helper.make_tensor_value_info(
                conf_name, onnx.TensorProto.FLOAT, [1, 1, H, W]
            ))
            
            print(f"  CLS stride {stride_idx}: reshape + sigmoid + reducesum")
    
    # Step 5: Build optimized graph
    print(f"\nStep 5: Building optimized graph...")
    kept_nodes = [n for n in model.graph.node if n.name not in nodes_to_remove]
    all_nodes = kept_nodes + new_nodes
    
    new_graph = helper.make_graph(
        all_nodes,
        model.graph.name,
        model.graph.input,
        new_outputs,
        model.graph.initializer
    )
    
    new_model = helper.make_model(new_graph, ir_version=model.ir_version)
    for opset in model.opset_import:
        opset.version = 12  # Force opset 12
        new_model.opset_import.append(opset)
    
    # Validate
    print("Validating...")
    try:
        new_model = shape_inference.infer_shapes(new_model)
        onnx.checker.check_model(new_model)
        print("✓ Validation passed")
    except Exception as e:
        print(f"⚠ Validation warning: {e}")
    
    # Save
    print(f"\nSaving to {onnx_path}...")
    onnx.save(new_model, onnx_path)
    
    # Cleanup
    if os.path.exists(temp_onnx):
        os.remove(temp_onnx)
    
    print(f"\n✓ Optimized model saved!")
    print(f"  Nodes: {len(all_nodes)} (removed {len(nodes_to_remove)}, added {len(new_nodes)})")
    print(f"  Outputs: {len(new_outputs)} (DFL + class sigmoid + confidence)")
    
    return True

if __name__ == '__main__':
    pt_path = sys.argv[1] if len(sys.argv) > 1 else 'yolo11n.pt'
    onnx_path = sys.argv[2] if len(sys.argv) > 2 else 'yolo11n_optimized.onnx'
    
    success = export_optimized(pt_path, onnx_path)
    sys.exit(0 if success else 1)
