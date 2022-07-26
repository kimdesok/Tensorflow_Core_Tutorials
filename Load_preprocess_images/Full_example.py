import pandas as pd
import tensorflow as tf

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

'''
1. #3 (age)
2. #4 (sex)
3. #9 (cp) [cp: chest pain type]
    -- Value 1: typical angina
    -- Value 2: atypical angina
    -- Value 3: non-anginal pain
    -- Value 4: asymptomatic
4. #10 (trestbps) resting blood pressure (in mm Hg on admission to the hospital)
5. #12 (chol) serum cholestoral in mg/dl
6. #16 (fbs) fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
7. #19 (restecg)  resting electrocardiographic results
    -- Value 0: normal
    -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
8. #32 (thalach)  maximum heart rate achieved
9. #38 (exang) exercise induced angina (1 = yes; 0 = no)
10. #40 (oldpeak)  ST depression induced by exercise relative to rest
11. #41 (slope) the slope of the peak exercise ST segment
    -- Value 1: upsloping
    -- Value 2: flat
    -- Value 3: downsloping
12. #44 (ca)  number of major vessels (0-3) colored by flourosopy
13. #51 (thal) 3 = normal; 6 = fixed defect; 7 = reversable defect
14. #58 (num) (the predicted attribute)
 diagnosis of heart disease (angiographic disease status)
    -- Value 0: < 50% diameter narrowing
    -- Value 1: > 50% diameter narrowing
(in any major vessel)

In this dataset some of the "integer" features in the raw data are actually categorical indices. 
These indices are not ordered numeric values.  Unordered numeric values are inappropriate to feed directly to the model; 
the model would interpret them as being ordered. 

To use these inputs you'll need to encode them, either as one-hot vectors or embedding vectors. 
The same applies to string-categorical features.
'''
df = pd.read_csv(csv_file)
print(df.info())
print(df.head())
print(df.dtypes)

target = df.pop('target')

# Full example

# build the preprocessing head
# Binary features do not need to be encoded or normalized
binary_feature_names = ['sex', 'fbs', 'exang']
categorical_feature_names = ['cp', 'restecg', 'slope', 'thal', 'ca']

inputs ={}
for name, column in df.items():
    if type(column[0]) == str:
        dtype = tf.string
    elif (name in categorical_feature_names or
          name in binary_feature_names):
        dtype = tf.int64
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

print(inputs)

# binary inputs
preprocessed = []
for name in binary_feature_names:
    inp = inputs[name]
    inp = inp[:, tf.newaxis]
    float_value = tf.cast(inp, tf.float32)
    preprocessed.append(float_value)
print(preprocessed)

def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


numeric_feature_names = ['age', 'thalach', 'trestbps', 'chol', 'oldpeak']
numeric_features = df[numeric_feature_names]

# numeric inputs
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

numeric_inputs = {}
for name in numeric_feature_names:
    numeric_inputs[name]=inputs[name]
numeric_inputs = stack_dict(numeric_inputs)
numeric_normalized = normalizer(numeric_inputs)
preprocessed.append(numeric_normalized)
print(preprocessed)

# categorical features
# example
vocab = ['a', 'b', 'c']
lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
print(lookup(['c','a','a','b','zzz']))

vocab = [1,4,7,99]
lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

print(lookup([-1,4,1]))

# create a layer that converts the vocabulary to a one hot vector

for name in categorical_feature_names:
    vocab = sorted(set(df[name]))
    print(f'name: {name}')
    print(f'vocab: {vocab}\n')
    if type(vocab[0]) is str:
        lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
    else:
        lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

    x = inputs[name][:, tf.newaxis]
    x = lookup(x)
    print(preprocessed.append(x))

# Assemble the preprocessing head

print(preprocessed)

# Concatenate all the preprocessed features along the depth axis to make a single vector
preprocessed_result = tf.concat(preprocessed, axis=-1)
print(preprocessed_result)

preprocessor = tf.keras.Model(inputs, preprocessed_result)

tf.keras.utils.plot_model(preprocessor, rankdir='LR', show_shapes=True)

# Test the preprocessor, slicing the first example from the DataFrame
print(preprocessor(dict(df.iloc[:1])))

# Create and train a model
body = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

print(inputs)

x = preprocessor(inputs)
print(x)

result = body(x)
print(result)

model = tf.keras.Model(inputs, result)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(dict(df), target, epochs=5, batch_size=BATCH_SIZE)

ds = tf.data.Dataset.from_tensor_slices((
    dict(df),
    target
))

ds = ds.batch(BATCH_SIZE)

import pprint
for x,y in ds.take(1):
    pprint.pprint(x)
    print()
    print(y)
history = model.fit(ds, epochs=5)
