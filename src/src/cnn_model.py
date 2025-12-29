from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

class CNNModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    def load_model(self, filepath):
        self.model = load_model(filepath)
