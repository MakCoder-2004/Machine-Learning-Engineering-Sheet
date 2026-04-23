import numpy as np
import tensorflow as tf

# Building Model
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])  
])

# Compiling Model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Data
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Training the model on the data
model.fit(xs, ys, epochs=500)

# Predicting
print(model.predict([10.0]))

# Saving the model
import time
saved_model_path = "./{}.h5".format(int(time.time()))

model.save(saved_model_path)

# Converting Model
!tensorflowjs_converter --input_format=keras {saved_model_path} ./