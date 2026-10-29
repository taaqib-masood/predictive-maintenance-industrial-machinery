class MetaController:
    def __init__(self, gbm_model, lstm_model, cnn_model):
        self.gbm_model = gbm_model
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model
    
    def predict(self, X_gbm, X_lstm, X_cnn, confidence_weights=None):
        y_gbm = self.gbm_model.predict(X_gbm)
        y_lstm = self.lstm_model.predict(X_lstm).flatten()
        y_cnn = self.cnn_model.predict(X_cnn).flatten()
        
        if confidence_weights is None:
            confidence_weights = [1/3, 1/3, 1/3]
        
        y_meta = (confidence_weights[0]*y_gbm +
                  confidence_weights[1]*y_lstm +
                  confidence_weights[2]*y_cnn)
        return y_meta
