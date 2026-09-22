#!/usr/bin/env python3
"""
优化 yolo26n-depth ONNX for RV1126B RKNN conversion.
剥离深度解码后处理 (Clip, Exp, Pow, Mul, Resize)，保留原始深度预测。
"""
import onnx
from onnx import helper, shape_inference, TensorProto

def optimize():
    model = onnx.load('yolo26n-depth.onnx')
    model = shape_inference.infer_shapes(model)
    g = model.graph
    
    print("=" * 70)
    print("优化 yolo26n-depth ONNX")
    print("=" * 70)
    print(f"原始节点数: {len(g.node)}")
    
    # 找到 head.head.3/Conv 的输出
    depth_pred_output = None
    for node in g.node:
        if node.name == '/model.23/head/head.3/Conv':
            depth_pred_output = node.output[0]
            break
    
    if not depth_pred_output:
        raise RuntimeError("找不到 head.head.3/Conv")
    
    print(f"深度预测输出: {depth_pred_output}")
    
    # 获取形状
    depth_shape = None
    for vi in g.value_info:
        if vi.name == depth_pred_output:
            depth_shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            break
    print(f"深度预测形状: {depth_shape}")
    
    # 反向追踪保留节点
    output_to_node = {}
    for node in g.node:
        for out in node.output:
            output_to_node[out] = node
    
    visited_outputs = set()
    visited_nodes = set()
    to_visit = [depth_pred_output]
    
    input_names = {inp.name for inp in g.input}
    init_names = {init.name for init in g.initializer}
    
    while to_visit:
        current = to_visit.pop()
        if current in visited_outputs:
            continue
        visited_outputs.add(current)
        
        if current in input_names or current in init_names:
            continue
        
        if current in output_to_node:
            node = output_to_node[current]
            visited_nodes.add(node.name)
            for inp in node.input:
                to_visit.append(inp)
    
    kept_nodes = [n for n in g.node if n.name in visited_nodes]
    removed = len(g.node) - len(kept_nodes)
    print(f"保留节点: {len(kept_nodes)}, 移除: {removed}")
    
    # 新输出
    new_output = helper.make_tensor_value_info(
        depth_pred_output, TensorProto.FLOAT, depth_shape)
    
    new_graph = helper.make_graph(
        kept_nodes, g.name, g.input, [new_output], g.initializer)
    
    new_model = helper.make_model(new_graph, ir_version=model.ir_version)
    while len(new_model.opset_import) > 0:
        new_model.opset_import.pop()
    op = new_model.opset_import.add()
    op.domain = ""
    op.version = 12
    
    out_path = 'yolo26n-depth_optimized.onnx'
    onnx.save(new_model, out_path)
    print(f"保存: {out_path}")
    
    # 提取 cal_a, cal_b 参数 (深度解码需要)
    for init in g.initializer:
        if 'cal_a' in init.name or 'cal_b' in init.name:
            import numpy as np
            vals = np.frombuffer(init.raw_data, dtype=np.float32)
            print(f"  参数 {init.name}: {vals}")

if __name__ == '__main__':
    optimize()
