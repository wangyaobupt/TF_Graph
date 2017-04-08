import tensorflow as tf
import numpy as np

# generate data and labels,
# the target value is defined as 1/(1+exp(1)),
# Thus, this is a regression issue, fitting a curve satisfying w1*x1 + w2*x2 + .. wnXn + bias = -1
def generateDataAndLabels(batchSize, numOfDims):
  testData = np.random.randn(batchSize, numOfDims)
  #print "TestData = ", testData[0]
  labels = np.ones([batchSize, 1])* (1.0/(1.0 + np.exp(1)))
  #print "Label = ", labels[0]
  return [testData, labels]

if __name__ == "__main__":
  numberOfInputDims = 2
  batchSize = 1
  log_path =  "tf_writer"

  [input,labels] = generateDataAndLabels(batchSize, numberOfInputDims)

  inputTensor = tf.placeholder(tf.float32, [None, numberOfInputDims], name='inputTensor')
  labelTensor=tf.placeholder(tf.float32, [None, 1], name='LabelTensor')
  W = tf.Variable(tf.random_uniform([numberOfInputDims, 1], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')
  a = tf.nn.sigmoid(tf.matmul(inputTensor, W) + b, name='activation')

  loss = tf.nn.l2_loss(a - labels, name='L2Loss')
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  #print "Weight = ", sess.run(W)
  #print "Bias = ", sess.run(b)
  #print 'Loss =', sess.run(loss, feed_dict={inputTensor:input})
  sess.run(train_step, feed_dict={inputTensor: input, labelTensor:labels})
  writer = tf.summary.FileWriter(log_path, sess.graph)
  writer.close()

  sess.close()


