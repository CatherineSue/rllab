import tensorflow as tf
# Build your graph.
#sc = tf.VariableScope(True, "foo")
#with tf.variable_scope(sc):
with tf.variable_scope("foo"):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

#with tf.variable_scope(sc):
with tf.variable_scope("foo", reuse=True):
    c = tf.matmul(a, b, name='c')

with tf.Session() as sess:
    # `sess.graph` provides access to the graph used in a <a href="../api_docs/python/tf/Session"><code>tf.Session</code></a>.
    writer = tf.summary.FileWriter("/tmp/log/", sess.graph)
    # Perform your computation...
    sess.run(c)
    #sess.run(d)
    # ...
    writer.close()
