import tensorflow as tf
import numpy as np
import os
import os.path
from tensorflow.contrib.tensorboard.plugins import projector

# generate data and labels,
# the target value is defined as 1/(1+exp(1)),
# Thus, this is a regression issue, fitting a curve satisfying w1*x1 + w2*x2 + .. wnXn + bias = -1
def generateDataAndLabels(batchSize, numOfDims):
  testData = np.random.randn(batchSize, numOfDims)
  #print "TestData = ", testData[0]
  labels = np.ones([batchSize, 1])
  for sampleIdx in range(batchSize):
    if np.argmax(testData[sampleIdx]) == 0:
      labels[sampleIdx] = 1
    else:
      labels[sampleIdx] = 0
  #print "Label = ", labels[0]
  return testData, labels

def removeFileInDir(targetDir): 
  for file in os.listdir(targetDir): 
    targetFile = os.path.join(targetDir,  file) 
    if os.path.isfile(targetFile):
      print 'Delete Old Log FIle:', targetFile
      os.remove(targetFile)

if __name__ == "__main__":
  numberOfInputDims = 2
  batchSize = 1000
  iterationNumber = 1000
  log_path =  "tf_writer"
  model_path = './singleNerualNode'

  removeFileInDir(log_path)

  [inputData,labels] = generateDataAndLabels(batchSize, numberOfInputDims)
  combinedDataAndLabel = np.column_stack((inputData, labels))
  np.savetxt('trainData.csv', combinedDataAndLabel , delimiter=",")
  
  embedding_var = tf.Variable(inputData, 'data_embeding')
  config = projector.ProjectorConfig()
  embedding = config.embeddings.add()
  embedding.tensor_name = embedding_var.name
  projector.visualize_embeddings(tf.summary.FileWriter(log_path), config)

  inputTensor = tf.placeholder(tf.float32, [None, numberOfInputDims], name='inputTensor')
  labelTensor = tf.placeholder(tf.float32, [None, 1], name='LabelTensor')
  with tf.name_scope('Nerual_Node'):
    W = tf.Variable(tf.random_normal([numberOfInputDims, 1]), name='weights')
    tf.summary.histogram('weights', W)
    b = tf.Variable(tf.zeros([1]), name='biases')
    tf.summary.histogram('biases', b)
    a = tf.nn.sigmoid(tf.matmul(inputTensor, W) + b, name='activation')

  with tf.name_scope('evaluation'):
    loss = tf.nn.l2_loss(a - labels, name='L2Loss') / batchSize
    threshold = 0.5 
    binary_outputs = a >= threshold
    binary_labels = labels >= threshold
    correct_item = tf.equal(binary_outputs, binary_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_item, tf.float32))
  tf.summary.scalar('L2Loss',loss)
  tf.summary.scalar('Accuracy', accuracy)
  
  train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
  
  merged = tf.summary.merge_all()

  all_vars = tf.global_variables()
  saver = tf.train.Saver(all_vars)

  sess = tf.Session()
  writer = tf.summary.FileWriter(log_path, sess.graph)
  
  sess.run(tf.global_variables_initializer())
  
  for iterIdx in range(iterationNumber):
    sess.run(train_step, feed_dict={inputTensor: inputData, labelTensor:labels})
    summary = sess.run(merged, feed_dict={inputTensor: inputData, labelTensor:labels})
    writer.add_summary(summary, iterIdx)
    if iterIdx %50 == 0:
        writer.flush()
    saver.save(sess, model_path)
  writer.close()

  sess.close()


