import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
class ArcFace(Layer):
    def __init__(self, n_classes=10, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape)
#         print(input_shape[-1],input_shape)
        self.W = self.add_weight(name='W',
                                shape=(input_shape[-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x = inputs
        c = tf.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        return logits
        # add margin
        # clip logits to prevent zero division when backward

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)
    
def af_loss(y_true,y_pred, s=30.0, m=0.50,):
    y=y_true
    logits=y_pred
    m=0.5
    s=30.
    theta = tf.acos(tf.clip_by_value(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
    target_logits = tf.cos(theta + m)

    logits = logits * (1 - y) + target_logits * y
    # feature re-scale
    logits *= s
    out = tf.nn.softmax(logits)
    loss=tf.losses.categorical_crossentropy(y_true,out)
    return loss
if __name__=='__main__':
    def make_model():
        inp=Input((28,28,1))    
        x=inp
        x=Conv2D(32,3)(x)
        x=Activation('relu')(x)
        x=MaxPool2D()(x)

        x=Conv2D(64,3)(x)
        x=Activation('relu')(x)
        x=MaxPool2D()(x)


        X=Dropout(0.5)(x)
        x=Conv2D(128,3)(x)
        x=Activation('relu')(x)
        x=MaxPool2D()(x)
        x=Flatten()(x)

        x=Dense(3)(x)
        x = ArcFace(n_classes=10)(x)
        model=tf.keras.models.Model(inp,x)
        model.compile(optimizer=tf.optimizers.Adam(1e-3),loss=ac_loss,metrics=['accuracy'])
        return model
    model=make_model()