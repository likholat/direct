from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.strided_slice import StridedSlice
from mo.front.onnx.extractors.utils import onnx_attr
from extensions.ops.roll import Roll
from ...ops.FFT import FFT, IFFT


class FFTFrontReplacer(FrontReplacementOp):
    op = "FFT"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        centered = onnx_attr(node, "centered", "i", default=False)
        fft = FFT(graph, {"name": node.name + "/fft_"}).create_node()

        if not centered:
            node.in_port(0).get_connection().set_destination(fft.in_port(0))
            return [fft.id]
        else:
            dim = onnx_attr(node, "dim", "ints", default=[2, 3])

            # shapeof = Shape(graph, {'name': node.name + '/Shape'}).create_node([node])
            # slice = StridedSlice(graph, {'name': node.name + '/StridedSlice',
            #                             'begin_mask': [1,1,0,0,0],
            #                             'end_mask': [1,1,0,0,0],
            #                             'new_axis_mask': [0,0,0,0],
            #                             'shrink_axis_mask': [0,0,1,1,1],
            #                             'ellipsis_mask': [0,0,0,0,0]}).create_node()

            input_node = Node(graph, graph.get_nodes_with_attributes(op="Parameter")[0])
            shift_x = int(input_node.shape[1] / 2)
            shift_y = int(input_node.shape[2] / 2)

            shift_i = Const(graph, {"value": [shift_x, shift_y]}).create_node()
            axes_i = Const(graph, {"value": dim}).create_node()
            roll_ifftshift = Roll(graph, {"name": node.name + "/ifftshift_"}).create_node()

            shift = Const(graph, {"value": [-shift_x, -shift_y]}).create_node()
            axes = Const(graph, {"value": dim}).create_node()
            roll_fftshift = Roll(graph, {"name": node.name + "/fftshift_"}).create_node()

            # slice.in_port(0).connect(shapeof.out_port(0)) #[ ERROR ]: Graph contains a cycle. Can not proceed.

            node.in_port(0).get_connection().set_destination(roll_ifftshift.in_port(0))

            roll_ifftshift.in_port(1).connect(shift_i.out_port(0))
            roll_ifftshift.in_port(2).connect(axes_i.out_port(0))

            fft.in_port(0).connect(roll_ifftshift.out_port(0))

            roll_fftshift.in_port(0).connect(fft.out_port(0))
            roll_fftshift.in_port(1).connect(shift.out_port(0))
            roll_fftshift.in_port(2).connect(axes.out_port(0))

            return [roll_fftshift.id]


class IFFTFrontReplacer(FrontReplacementOp):
    op = "IFFT"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        centered = onnx_attr(node, "centered", "i", default=False)
        ifft = IFFT(graph, {"name": node.name + "/ifft_", "inverse": 1}).create_node()

        if not centered:
            node.in_port(0).get_connection().set_destination(ifft.in_port(0))
            return [ifft.id]
        else:
            dim = onnx_attr(node, "dim", "ints", default=[2, 3])
            input_node = Node(graph, graph.get_nodes_with_attributes(op="Parameter")[0])
            shift_x = int(input_node.shape[1] / 2)
            shift_y = int(input_node.shape[2] / 2)

            shift_i = Const(graph, {"value": [shift_x, shift_y]}).create_node()
            axes_i = Const(graph, {"value": dim}).create_node()
            roll_ifftshift = Roll(graph, {"name": node.name + "/ifftshift_"}).create_node()

            shift = Const(graph, {"value": [-shift_x, -shift_y]}).create_node()
            axes = Const(graph, {"value": dim}).create_node()
            roll_fftshift = Roll(graph, {"name": node.name + "/fftshift_"}).create_node()

            node.in_port(0).get_connection().set_destination(roll_ifftshift.in_port(0))
            roll_ifftshift.in_port(1).connect(shift_i.out_port(0))
            roll_ifftshift.in_port(2).connect(axes_i.out_port(0))

            ifft.in_port(0).connect(roll_ifftshift.out_port(0))

            roll_fftshift.in_port(0).connect(ifft.out_port(0))
            roll_fftshift.in_port(1).connect(shift.out_port(0))
            roll_fftshift.in_port(2).connect(axes.out_port(0))

            return [roll_fftshift.id]