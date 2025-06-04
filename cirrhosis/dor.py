#!/usr/bin/env python3
"""
lean_cirrhosis_dsn_pipeline.py

A lighter Variable Selection Network for faster training:
- 2 VSFlow stages (16→8 units)
- Dropouts 0.5→0.25
- Batch size 128, epochs 50
- tf.data.Dataset for pipelined I/O

Fixed version with thorough data cleaning and debugging
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

#─────────────────────────────────────────────────────────────────────────────
# Custom Layers
#─────────────────────────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
def smish(x):
    return x * tf.keras.backend.tanh(tf.keras.backend.log(1 + tf.keras.backend.sigmoid(x)))

class GatedLinearUnit(L.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.dense_linear = L.Dense(units)
        self.dense_gate   = L.Dense(units, activation='sigmoid')
    
    def call(self, inputs):
        return self.dense_linear(inputs) * self.dense_gate(inputs)

class GatedResidualNetwork(L.Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense_relu = L.Dense(units, activation=smish)
        self.dense_linear = L.Dense(units)
        self.dropout = L.Dropout(dropout_rate)
        self.glu = GatedLinearUnit(units)
        self.layer_norm = L.LayerNormalization()
        self.project = L.Dense(units)
    
    def call(self, x):
        res = x
        x = self.dense_relu(x)
        x = self.dense_linear(x)
        x = self.dropout(x)
        if tf.keras.backend.int_shape(res)[-1] != self.units:
            res = self.project(res)
        x = res + self.glu(x)
        return self.layer_norm(x)

class VariableSelection(L.Layer):
    def __init__(self, num_features, units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.grns = [GatedResidualNetwork(units, dropout_rate) for _ in range(num_features)]
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = L.Dense(num_features, activation='softmax')
    
    def call(self, inputs_list):
        cat = tf.concat(inputs_list, axis=-1)
        w = self.softmax(self.grn_concat(cat))
        w = tf.expand_dims(w, -1)
        stacked = tf.stack([self.grns[i](inputs_list[i]) for i in range(len(inputs_list))], axis=1)
        out = tf.squeeze(tf.matmul(w, stacked, transpose_a=True), axis=1)
        return out

class VariableSelectionFlow(L.Layer):
    def __init__(self, num_features, units, dropout_rate, dense_units=None, **kwargs):
        super().__init__(**kwargs)
        self.split = L.Lambda(lambda t: tf.split(t, num_features, axis=-1))
        self.vs = VariableSelection(num_features, units, dropout_rate)
        self.denses = [L.Dense(dense_units) for _ in range(num_features)] if dense_units else None

    def call(self, x):
        parts = self.split(x)
        if self.denses:
            parts = [self.denses[i](parts[i]) for i in range(len(parts))]
        return self.vs(parts)

#─────────────────────────────────────────────────────────────────────────────
# Load and Clean Data
#─────────────────────────────────────────────────────────────────────────────
def clean_column(series, col_name):
    if col_name == 'Drug':
        series = series.map({'Placebo': 0, 'D-penicillamine': 1}).fillna(0)
    elif col_name == 'Sex':
        series = series.map({'male': 0, 'female': 1, 'M': 0, 'F': 1}).fillna(0)
    elif col_name in ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema']:
        series = series.map({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, 'y': 1, 'n': 0, 'yes': 1, 'no': 1, 'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0)
    return pd.to_numeric(series, errors='coerce').fillna(0)

train = pd.read_csv('cirrhosis/train.csv')
test = pd.read_csv('cirrhosis/test.csv')

status_mapping = {'C': 0, 'CL': 1, 'D': 2}
y = train['Status'].map(status_mapping).fillna(0).astype(int).values
target = to_categorical(y, num_classes=3).astype(np.float32)
train = train.drop(columns=['Status'])

common_cols = [col for col in train.columns if col in test.columns]

if 'id' in test.columns:
    test_ids = test['id'].copy()
    if 'id' in common_cols:
        common_cols.remove('id')
else:
    raise ValueError("Test set must include an 'id' column.")

for col in common_cols:
    train[col] = clean_column(train[col], col).astype(np.float32)
    test[col] = clean_column(test[col], col).astype(np.float32)

X_train = train[common_cols].values.astype(np.float32)
X_test = test[common_cols].values.astype(np.float32)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

split = int(X_train.shape[0] * 0.9)
X_tr, X_val = X_train[:split], X_train[split:]
y_tr, y_val = target[:split], target[split:]

def make_dataset(X, y, batch_size, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(X, tf.float32), tf.cast(y, tf.float32)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_tr, y_tr, 128, shuffle=True)
val_ds = make_dataset(X_val, y_val, 128)

#─────────────────────────────────────────────────────────────────────────────
# Build Model (after feature fix)
#─────────────────────────────────────────────────────────────────────────────
n_features = len(common_cols)
print(f"\nBuilding model with {n_features} features...")

inputs = Input(shape=(n_features,), dtype=tf.float32)
x = VariableSelectionFlow(n_features, units=16, dropout_rate=0.5, dense_units=4)(inputs)
x = VariableSelectionFlow(16, units=8, dropout_rate=0.25)(x)
out = L.Dense(3, activation='softmax')(x)

model = Model(inputs, out)
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#─────────────────────────────────────────────────────────────────────────────
# Train
#─────────────────────────────────────────────────────────────────────────────
es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[es, lr], verbose=2)

#─────────────────────────────────────────────────────────────────────────────
# Predict & Save Submission
#─────────────────────────────────────────────────────────────────────────────
print("\nMaking predictions...")
probs = model.predict(X_test, batch_size=128)

submission = pd.DataFrame(
    probs,
    index=test_ids,
    columns=['Status_C', 'Status_CL', 'Status_D']
)
submission.index.name = 'id'
submission.to_csv('cirrhosis/submission.csv')

print('✅ Saved submission.csv successfully.')
print(f"📐 Submission shape: {submission.shape}")
print(f"🔍 Sample predictions:\n{submission.head()}")