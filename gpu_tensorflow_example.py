import tensorflow as tf

# Check if a GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU found. TensorFlow will use GPU.")
else:
    print("No GPU found. TensorFlow will use CPU.")

# Create a simple TensorFlow operation
a = tf.constant(2.0)
b = tf.constant(3.0)
result = tf.multiply(a, b)

# Display the result
print("Result:", result.numpy())

