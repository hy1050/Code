import tensorflow as tf
from tensorflow.keras import activations, layers, models
import matplotlib.pyplot as plt
import os

TF_ENABLE_ONEDNN_OPTS=0
PATCH_SIZE = 5
NUM_CLASSES = 43
NUM_EPOCH = 10
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
BATCH_SIZE = 128


def params():
    # weights and biases
    params = {}

    params['w_conv1'] = tf.compat.v1.get_variable(
        'w_conv1',
        shape=[PATCH_SIZE, PATCH_SIZE, 3, 32],
        initializer=tf.keras.initializers.GlorotUniform())
    params['b_conv1'] = tf.Variable(tf.constant(0.1, shape=[32]))

    params['w_conv2'] = tf.compat.v1.get_variable(
        'w_conv2',
        shape=[PATCH_SIZE, PATCH_SIZE, 32, 64],
        initializer=tf.keras.initializers.GlorotUniform())
    params['b_conv2'] = tf.Variable(tf.constant(0.1, shape=[64]))

    params['w_conv3'] = tf.compat.v1.get_variable(
        'w_conv3',
        shape=[PATCH_SIZE, PATCH_SIZE, 64, 128],
        initializer=tf.keras.initializers.GlorotUniform())
    params['b_conv3'] = tf.Variable(tf.constant(0.1, shape=[128]))

    params['w_fc1'] = tf.compat.v1.get_variable(
        'w_fc1',
        shape=[4 * 4 * 128, 2048],
        initializer=tf.keras.initializers.GlorotUniform())
    params['b_fc1'] = tf.Variable(tf.constant(0.1, shape=[2048]))

    params['w_fc2'] = tf.compat.v1.get_variable(
        'w_fc2',
        shape=[2048, NUM_CLASSES],
        initializer=tf.keras.initializers.GlorotUniform())
    params['b_fc2'] = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))

    return params


def cnn(data, model_params, keep_prob):
    # First layer
    h_conv1 = tf.keras.layers.Conv2D(
            filters=data, 
            kernel_size=model_params['w_conv1'],
            strides=(1, 1), 
            padding='SAME', 
            activation='relu',
            use_bias=True, 
            bias_initializer=tf.keras.initializers.Constant(value=model_params['b_conv1']))
    h_pool1 = layers.MaxPooling2D(
        h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second layer
    h_conv2 = activations.relu(
        layers.Conv2D(
            h_pool1, model_params['w_conv2'], [1, 1, 1, 1], padding='SAME') +
        model_params['b_conv2'])
    h_pool2 = layers.MaxPooling2D(
        h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Third layer
    h_conv3 = activations.relu(
        layers.Conv2D(
            h_pool2, model_params['w_conv3'], [1, 1, 1, 1], padding='SAME') +
        model_params['b_conv3'])
    h_pool3 = layers.MaxPooling2D(
        h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    conv_layer_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
    h_fc1 = activations.relu(
        tf.matmul(conv_layer_flat, model_params['w_fc1']) +
        model_params['b_fc1'])
    h_fc1 = layers.Dropout(h_fc1, keep_prob)

    # Output layer
    out = tf.matmul(h_fc1, model_params['w_fc2']) + model_params['b_fc2']

    return out


class TrafficSignRecognizer:
    def __init__(self, mode, model_dir):
        assert mode in {'train', 'inference'}
        self.mode = mode
        self.model_dir = model_dir
        self.recognizer_model = self.build(mode)

    def build(self, mode):
        assert mode in {'train', 'inference'}
        input_image = tf.keras.layers.Input(
            shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        x = tf.keras.layers.Conv2D(
            32, PATCH_SIZE, padding='same', activation='relu')(input_image)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(
            64, PATCH_SIZE, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(
            128, PATCH_SIZE, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4*4*128, activation='relu')(x)
        if mode == 'train':
            x = tf.keras.layers.Dropout(0.5)(x, training=True)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        return tf.keras.Model(inputs=input_image, outputs=outputs)

    def compile(self, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.recognizer_model.compile(
            optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_dataset, train_labels, learning_rate):
        assert self.mode == 'train', 'Create model in train mode'

        # Callbacks
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(
                self.model_dir, 'log_dir'), histogram_freq=0, write_graph=True),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, 'ckpt'), verbose=0, save_weights_only=True),
        ]

        # Compile
        self.compile(learning_rate)

        # Do training
        History = self.recognizer_model.fit(
            train_dataset, train_labels, callbacks=callbacks, epochs=NUM_EPOCH, validation_split=0.2)
        
        # Save
        tf.keras.models.save_model(self.recognizer_model, 'my_model_cnn.h5')
        plt.plot(History.history['accuracy'])
        plt.plot(History.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('accuracy_plot.png')

        # Biểu đồ hàm mất mát và lưu vào file
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('loss_plot.png')
        
if __name__ == '__main__':
    print('For training')
    tsr = TrafficSignRecognizer(mode='train', model_dir='train_logs')
    tsr.recognizer_model.summary()

    print('For inference')
    tsr_inference = TrafficSignRecognizer(
        mode='inference', model_dir='inference_logs')
    tsr_inference.recognizer_model.summary()
