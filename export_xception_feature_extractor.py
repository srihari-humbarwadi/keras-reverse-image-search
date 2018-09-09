from keras.models import Model, load_model

model = load_model('xception.h5')
model.layers.pop(-1)
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

model.save('xception_feature_extractor.h5')