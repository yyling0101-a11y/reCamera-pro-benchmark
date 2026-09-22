#!/usr/bin/env python3
"""
YOLO26 ONNX optimizer for RV1126B.
YOLO26 has NO DFL - uses 4-channel direct bbox regression instead of 64-channel DFL.
This optimizer strips the decode/postprocessing and outputs stride-level spatial features.
"""
import os, sys
import onnx
from onnx import shape_inference, helper, numpy_helper, TensorProto
import numpy as np


def find_downstream_nodes(graph, seed_tensors):
    """Find all nodes downstream of seed tensors (BFS)."""
    tensor_consumers = {}
    for node in graph.node:
        for inp in node.input:
            tensor_consumers.setdefault(inp, []).append(node.name)
    nodes_to_remove = set()
    queue = list(seed_tensors)
    while queue:
        tensor = queue.pop(0)
        for node_name in tensor_consumers.get(tensor, []):
            if node_name not in nodes_to_remove:
                nodes_to_remove.add(node_name)
                for node in graph.node:
                    if node.name == node_name:
                        queue.extend(node.output)
    return nodes_to_remove


def find_stride_reshapes(graph):
    """Find Reshape nodes in head that feed Concat (stride-level outputs)."""
    tensor_shapes = {}
    for vi in graph.value_info:
        if vi.type.tensor_type.HasField('shape'):
            shape = [d.dim_value if d.dim_value > 0 else None
                     for d in vi.type.tensor_type.shape.dim]
            tensor_shapes[vi.name] = shape
    node_outputs = {}
    for node in graph.node:
        for o in node.output:
            node_outputs[o] = node

    reshapes = []
    for node in graph.node:
        if node.op_type != 'Concat' or '/model.23/' not in node.name:
            continue
        if len(node.input) != 3:
            continue
        all_from_reshape = all(
            node_outputs.get(inp, None) is not None and node_outputs[inp].op_type == 'Reshape'
            for inp in node.input
        )
        if not all_from_reshape:
            continue
        for inp in node.input:
            shape = tensor_shapes.get(inp)
            if shape and len(shape) == 3:
                reshapes.append({'name': inp, 'channels': shape[1], 'spatial': shape[2]})
    return reshapes


def classify_branches_yolo26(reshapes, task_type):
    """Classify reshape branches for YOLO26 (no DFL, 4ch bbox)."""
    groups = {}
    for r in reshapes:
        groups.setdefault(r['channels'], []).append(r)
    for ch in groups:
        groups[ch].sort(key=lambda x: x['spatial'], reverse=True)

    branches = {}
    if task_type == 'det':
        if 4 in groups:
            branches['bbox'] = groups[4]
        if 80 in groups:
            branches['cls'] = groups[80]
    elif task_type == 'seg':
        if 4 in groups:
            branches['bbox'] = groups[4]
        if 80 in groups:
            branches['cls'] = groups[80]
        if 32 in groups:
            branches['mask_coeff'] = groups[32]
    elif task_type == 'pose':
        for ch in sorted(groups.keys(), reverse=True):
            if ch == 4:
                branches['bbox'] = groups[ch]
            elif ch == 1:
                branches['objectness'] = groups[ch]
            elif ch == 80:
                branches['cls'] = groups[ch]
            else:
                branches['keypoint'] = groups[ch]
    elif task_type == 'obb':
        for ch in sorted(groups.keys(), reverse=True):
            if ch == 4:
                branches['bbox'] = groups[ch]
            elif ch == 1:
                branches['angle'] = groups[ch]
            elif ch >= 15:  # DOTA has 15 classes
                branches['cls'] = groups[ch]
    return branches


def add_spatial_outputs(model, branches, stride_sizes):
    """Add spatial Reshape (+ optional Sigmoid) outputs for each branch."""
    new_nodes, new_outputs = [], []
    for branch_name, reshapes in branches.items():
        for i, r in enumerate(reshapes):
            C = r['channels']
            H, W = stride_sizes[i]
            spatial_shape = [1, C, H, W]
            reshape_name = f'{branch_name}_stride_{i}_spatial'
            shape_tensor = numpy_helper.from_array(
                np.array(spatial_shape, dtype=np.int64),
                name=f'{branch_name}_stride_{i}_shape_init')
            model.graph.initializer.append(shape_tensor)
            new_nodes.append(helper.make_node(
                'Reshape', inputs=[r['name'], shape_tensor.name],
                outputs=[reshape_name], name=f'reshape_{branch_name}_{i}'))

            if branch_name == 'cls':
                # Apply sigmoid to class scores
                sigmoid_name = f'{branch_name}_stride_{i}_sigmoid'
                new_nodes.append(helper.make_node(
                    'Sigmoid', inputs=[reshape_name],
                    outputs=[sigmoid_name], name=f'sigmoid_{branch_name}_{i}'))
                new_outputs.append(helper.make_tensor_value_info(
                    sigmoid_name, TensorProto.FLOAT, spatial_shape))
            else:
                new_outputs.append(helper.make_tensor_value_info(
                    reshape_name, TensorProto.FLOAT, spatial_shape))
    return new_nodes, new_outputs


def optimize_onnx(onnx_path, output_path, task_type):
    print(f"\nOptimizing YOLO26: {onnx_path} ({task_type})")
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)
    print(f"Original: {len(model.graph.node)} nodes")

    reshapes = find_stride_reshapes(model.graph)
    print(f"Found {len(reshapes)} stride reshapes")
    for r in reshapes:
        print(f"  ch={r['channels']}, spatial={r['spatial']}, name={r['name']}")

    branches = classify_branches_yolo26(reshapes, task_type)
    print(f"Branches: {list(branches.keys())}")
    for bname, breshapes in branches.items():
        print(f"  {bname}: {len(breshapes)} strides, ch={breshapes[0]['channels']}")

    stride_spatials = sorted(set(r['spatial'] for r in reshapes), reverse=True)
    stride_sizes = [(int(s**0.5), int(s**0.5)) for s in stride_spatials]
    print(f"Stride sizes: {stride_sizes}")

    seed_tensors = [r['name'] for r in reshapes]

    # Handle seg proto output
    proto_name, proto_shape = None, None
    if task_type == 'seg':
        for output in model.graph.output:
            if 'output1' in output.name or 'proto' in output.name.lower():
                proto_name = output.name
                proto_shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
                print(f"Proto output: {proto_name} shape={proto_shape}")

    # Find nodes to remove (downstream of seed tensors)
    nodes_to_remove = find_downstream_nodes(model.graph, seed_tensors)

    # Protect proto-related nodes for seg
    if proto_name:
        for node in model.graph.node:
            if proto_name in node.output or 'proto' in node.name:
                nodes_to_remove.discard(node.name)

    print(f"Removing {len(nodes_to_remove)} downstream nodes")

    # Add new spatial outputs
    new_nodes, new_outputs = add_spatial_outputs(model, branches, stride_sizes)

    # Keep proto output for seg
    if proto_name:
        new_outputs.append(helper.make_tensor_value_info(
            proto_name, TensorProto.FLOAT, proto_shape))

    # Build new graph
    kept_nodes = [n for n in model.graph.node if n.name not in nodes_to_remove]
    new_graph = helper.make_graph(
        kept_nodes + new_nodes, model.graph.name,
        model.graph.input, new_outputs, model.graph.initializer)
    new_model = helper.make_model(new_graph, ir_version=model.ir_version)

    # Set opset to 12 for RKNN compatibility
    while len(new_model.opset_import) > 0:
        new_model.opset_import.pop()
    _op = new_model.opset_import.add()
    _op.domain = ""
    _op.version = 12

    try:
        onnx.checker.check_model(shape_inference.infer_shapes(new_model))
        print("Validation: PASSED")
    except Exception as e:
        print(f"Warning: {e}")

    onnx.save(new_model, output_path)
    print(f"Saved: {output_path} ({len(kept_nodes)+len(new_nodes)} nodes, {len(new_outputs)} outputs)")
    for o in new_model.graph.output:
        shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"  {o.name}: {shape}")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: optimize_yolo26_onnx.py <in.onnx> <out.onnx> <det|seg|pose|obb>")
        sys.exit(1)
    optimize_onnx(sys.argv[1], sys.argv[2], sys.argv[3])
