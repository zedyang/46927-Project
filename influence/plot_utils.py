import os
from functools import partial
from io import BytesIO
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import PIL.Image
from IPython.display import clear_output, Image, display, HTML

import tensorflow as tf


# TODO: 那个"左右连线"的图，换model看influence怎么变
# TODO: LOO v.s. Influence
# TODO: brute-force v.s. approx
#

def strip_consts(graph_def, max_const_size=32):
    """
    Strip large constant values from graph_def.
    :param graph_def:
    :param max_const_size:
    :return:
    """
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes(
                    "<stripped %d bytes>" % size)
    return strip_def


def rename_nodes(graph_def, rename_func):
    """

    :param graph_def:
    :param rename_func:
    :return:
    """
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = (
                rename_func(s) if s[0] != '^' else '^' + rename_func(s[1:]))
    return res_def


def show_graph(graph_def, max_const_size=32):
    """
    Visualize TensorFlow graph.
    :param graph_def:
    :param max_const_size:
    :return:
    """
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="{url}" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(
        data=repr(str(strip_def)),
        id='graph' + str(np.random.rand()),
        url="https://tensorboard.appspot.com/tf-graph-basic.build.html"
    )

    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}">
        </iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
