"""
Written by Benjamin Raiford
8/8/2020
"""

import tensorflow as tf  # Using tensorflow==1.15.3

with tf.Session() as sess:
    # Restore variables
    saver = tf.train.import_meta_graph("tf1files/my_model.meta")
    saver.restore(sess, "tf1files/my_model")

    for operation in (sess.graph.get_operations()):
        print(operation.name)
        print(operation.values())

    """
    I only had a little bit of time to work on this,but below are functions I think might prove useful.
    
    sess.graph.get_tensor_by_name -- This should return a tensor if you can make the operations only tensors
        Currently there is an error using this because Placeholder is not a tensor... you'd have to be able to feed
        only tensors into this.
                    
    your_tensor_name.evaL() -- Supposedly, this will return your tensor as an array once you get the tensors
    
    Assuming you can run a loop where you get all the tensor names and then convert them to arrays using 
    your_tensor_name.eval(), then you could follow pseudocode along these lines (with model_as_array initialized 
    outside the scope, obviously.
    
        for(tensor in [however you manage to do this]):
            array_tensor = tensor.eval()
            model_as_array.append(array_tensor)
    """
