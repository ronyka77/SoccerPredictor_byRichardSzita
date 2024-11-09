from keras.callbacks import Callback
from keras.metrics import Metric
import keras.backend as K
import numpy as np
import cloudpickle as cp
from keras.models import load_model
from sklearn.base import BaseEstimator, RegressorMixin
import time

class WithinRangeMetric(Metric):
    def __init__(self, name='within_range_metric', **kwargs):
        super(WithinRangeMetric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        
        diff = K.abs(y_true - y_pred)
        within_range = K.less_equal(diff, 0.5)
        self.true_positives.assign_add(K.sum(K.cast(within_range, K.floatx())))
        self.total.assign_add(K.cast(K.shape(y_true)[0], K.floatx()))

    def result(self):
        return self.true_positives / self.total

    def reset_state(self):
        self.true_positives.assign(0)
        self.total.assign(0)

    def get_config(self):
        config = super(WithinRangeMetric, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CustomReduceLROnPlateau(Callback):
    def __init__(self, monitor='loss', factor=0.5, patience=10, verbose=0, min_lr=0.0001):
        super(CustomReduceLROnPlateau, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr

        self.best = None
        self.cooldown_counter = 0
        self.wait = 0
        self.monitor_op = lambda a, b: np.less(a, b)
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            print(f"Warning: CustomReduceLROnPlateau requires {self.monitor} to be available!")
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(K.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    K.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}.')
                self.wait = 0

    def get_config(self):
        return {
            'monitor': self.monitor,
            'factor': self.factor,
            'patience': self.patience,
            'verbose': self.verbose,
            'min_lr': self.min_lr,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CustomStackingRegressor:
    def __init__(self, stacking_regressor, keras_model, keras_model_path):
        self.stacking_regressor = stacking_regressor
        self.keras_model = keras_model
        self.keras_model_path = keras_model_path
    
    def save(self, model_path):
        self.keras_model.save(self.keras_model_path, include_optimizer=True)
        self.stacking_regressor.named_estimators_.pop('nn', None)
        with open(model_path, 'wb') as f:
            cp.dump(self.stacking_regressor, f)
        self.stacking_regressor.named_estimators_['nn'] = self.keras_model
    
    @classmethod
    def load(cls, model_path, keras_model_path, custom_objects=None):
        with open(model_path, 'rb') as f:
            stacking_regressor = cp.load(f)
        keras_model = load_model(keras_model_path, custom_objects=custom_objects)
        stacking_regressor.named_estimators_['nn'] = keras_model
        return cls(stacking_regressor, keras_model, keras_model_path)

class LoggingEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, name, logger):
        self.estimator = estimator
        self.name = name
        self.logger = logger

    def fit(self, X, y):
        self.logger.info(f"Fitting {self.name} started.")
        start_time = time.time()
        self.estimator.fit(X, y)
        end_time = time.time()
        self.logger.info(f"Fitting {self.name} completed in {end_time - start_time:.2f} seconds.")
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    @property
    def model_(self):
        return getattr(self.estimator, 'model_', None) 