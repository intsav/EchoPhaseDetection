from keras import layers
from keras import models
from keras.models import load_model
from keras import optimizers 
from keras import applications
from keras.layers import Input

def lstm_model():
    
    ''' ResNet50 for feature extraction '''
    input_tensor = Input((112,112,3))
    base_model = applications.resnet.ResNet50(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=True)
    
    # Encode feature map from final fc layer
    model = models.Model(inputs=base_model.input,
              outputs=base_model.get_layer('avg_pool').output) 
    
    # 2x LSTM network for temporal decoding
    lstm_input = layers.Input((30, 2048))
    input_shape = (30,2048)
    x = layers.LSTM(2048, return_sequences=True)(lstm_input)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.LSTM(512, return_sequences=True, dropout=0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(30)(x) # Regress 1 prediction per timestep
    
    # Combine both networks using Keras' Functional API
    # https://keras.io/guides/functional_api/
    lstm_model = models.Model(inputs=lstm_input, outputs=x, name="lstm_model")
    
    combined_input = layers.Input((30,112,112,3))
    encoded_img = layers.TimeDistributed(model)(combined_input) # TimeDistributed for video sequence data
    decoded_img = lstm_model(encoded_img)
    combined_arch = models.Model(inputs=combined_input, outputs=decoded_img, name="combined_model")

    # Set the metrics
    metrics = ['mse', 'mae']
    # Define the optimiser
    optimizer = optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # Set the loss function
    loss = 'mse'  
    # Compile the model
    combined_arch.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    # Return the combined architecture
    return combined_arch
