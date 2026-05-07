from .tracer import trace_context
from .pattern import PatternMatcher, FusionRegistry
from .graph import Graph,Node,NodeType
from .executor import GraphExecutor
from ..data import DataLoader
from ..base.functions import as_Tensor, quantize, dequantize, QuantizeRegistry, Quantize, Dequantize
import numpy as np

class GraphOptimizer:
    def __init__(self, model, sample_input):
        self.model = model
        self.sample_input = sample_input

    def origin_graph(self) -> Graph:
        # 记录一次前向，得到计算图
        with trace_context() as tracer:
            _ = self.model(self.sample_input)
            graph = tracer.get_graph()
        return graph

    def optimize(self) -> Graph:
        # 1. 记录一次前向，得到计算图
        with trace_context() as tracer:
            _ = self.model(self.sample_input)
            graph = tracer.get_graph()

        # 2. 匹配所有可融合模式
        matcher = PatternMatcher(graph, FusionRegistry)
        matches = matcher.find_all_matches()

        # 3. 依次替换子图
        for match in matches:
            match.replace(graph)

        return graph

    def optimize_to_executor(self) -> GraphExecutor:
        optimized_graph = self.optimize()
        return GraphExecutor(optimized_graph)
    
    def quantize_to_executor(self, cal_loader:DataLoader, dtype=np.int8) -> GraphExecutor:
        executor = self.optimize_to_executor() # 优化后的执行器
        executor.clear_record()

        # 校准
        for batch_idx, (Xb, yb) in enumerate(cal_loader):
            if Xb.shape[0] != cal_loader.batch_size:
                #print(f'batch {batch_idx} is not full. ({Xb.shape[0]}/{cal_loader.batch_size})')
                continue
            Xb = as_Tensor(Xb)
            _ = executor.forward_record(Xb)

        # 量化
        quantized_graph = apply_quantize(executor, dtype)
        quantized_executor = GraphExecutor(quantized_graph)
        return quantized_executor


def apply_quantize(executor:GraphExecutor, dtype):
    # 计算量化参数 (Max-Min) 并应用量化
    qmax = np.iinfo(dtype).max
    qmin = np.iinfo(dtype).min
    # 权重只需要静态替换
    for param_node in executor.param_nodes:
        rmax = param_node.rmax
        rmin = param_node.rmin
        scale = (rmax - rmin) / (qmax - qmin)
        zero_point = np.round(qmin - (rmin / scale))

        param_node.obj.data = quantize(param_node.obj.data, scale=scale, zero_point=zero_point, dtype=dtype)
    # 激活值需要添加Function节点
    graph = executor.graph
    for func_node in graph.func_nodes.values():
        # 需要量化
        if func_node in QuantizeRegistry.can_quantize:
            '''
            pre_nodes -> func_node
            变为
            pre_nodes -> quantize_func_nodes -> quantized_tensor_node -> func_node
            '''
            pre_nodes = graph.get_predecessors(func_node)
            graph._remove_edges_to_node(func_node, set(pre_nodes)) # 移除旧边
            for pre_node in pre_nodes:
                if pre_node.obj.dtype == dtype: # 已量化则跳过
                    graph.add_edge(pre_node, func_node) # 恢复被删除的边
                    continue
                rmax = pre_node.rmax
                rmin = pre_node.rmin
                scale = (rmax - rmin) / (qmax - qmin)
                zero_point = np.round(qmin - (rmin / scale))

                q_func = Quantize(scale=scale,zero_point=zero_point,dtype=dtype)
                q_tensor = q_func(pre_node.obj)

                quantize_func_node = graph.add_node(q_func)
                quantized_tensor_node = graph.add_node(q_tensor)
                graph.add_edge(pre_node, quantize_func_node)
                graph.add_edge(quantize_func_node, quantized_tensor_node)
                graph.add_edge(quantized_tensor_node, func_node)
        # 需要反量化
        elif func_node in QuantizeRegistry.risist_quantize:
            '''
            pre_nodes -> func_node
            变为
            pre_nodes -> dequantize_func_nodes -> dequantized_tensor_node -> func_node
            '''
            pre_nodes = graph.get_predecessors(func_node)
            graph._remove_edges_to_node(func_node, set(pre_nodes)) # 移除旧边
            for pre_node in pre_nodes:
                if pre_node.obj.dtype != dtype: # 未量化则跳过
                    graph.add_edge(pre_node, func_node) # 恢复被删除的边
                    continue
                rmax = pre_node.rmax
                rmin = pre_node.rmin
                scale = (rmax - rmin) / (qmax - qmin)
                zero_point = np.round(qmin - (rmin / scale))

                q_func = Dequantize(scale=scale,zero_point=zero_point)
                q_tensor = q_func(pre_node.obj)

                dequantize_func_node = graph.add_node(q_func)
                dequantized_tensor_node = graph.add_node(q_tensor)
                graph.add_edge(pre_node, dequantize_func_node)
                graph.add_edge(dequantize_func_node, dequantized_tensor_node)
                graph.add_edge(dequantized_tensor_node, func_node)
    return graph
