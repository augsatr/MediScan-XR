from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_model():

    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = base_model.output

    # Attention Layer
    gap = GlobalAveragePooling2D()(x)

    dense1 = Dense(512, activation='relu')(gap)
    dense2 = Dense(1024, activation='sigmoid')(dense1)

    attention = Reshape((1,1,1024))(dense2)

    x = Multiply()([x, attention])

    # Classification
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model