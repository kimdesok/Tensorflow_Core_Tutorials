import pandas as pd
import tensorflow as tf

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

df = pd.read_csv(csv_file)
print(df.info())
print(df.head())
print(df.dtypes)

target = df.pop('target')

#A dataframe as an array

numeric_feature_names = ['age', 'thalach', 'trestbps', 'chol', 'oldpeak']
numeric_features = df[numeric_feature_names]
print(numeric_features.head())

# Convert the dataframe to a numpy array
tf.convert_to_tensor(numeric_features)
print(tf.convert_to_tensor(numeric_features))

# Preparation for the model - normalize the feature array
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(numeric_features)
print(normalizer(numeric_features.iloc[:3]))

# Place the normilazation layer as the 1st layer

def get_basic_model():
  model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_basic_model()
model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)

# with Tf.data - numeric_features is the dataframe
numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))
for row in numeric_dataset.take(3):
    print(row)

numeric_batches = numeric_dataset.shuffle(1000).batch(BATCH_SIZE)
model = get_basic_model()
model.fit(numeric_batches, epochs=15)

# A dataframe as a dictionary - cast the dataframe to the dict.
numeric_dict_ds = tf.data.Dataset.from_tensor_slices((dict(numeric_features), target))

for row in numeric_dict_ds.take(3):
    print(row)

# Dictionaries with Keras

#1. The model-subclass style
def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)

class myModel(tf.keras.Model):
    def __init__(self):
        super().__init__(self)
        self.normalizer = tf.keras.layers.Normalization(axis=-1)

        self.seq = tf.keras.Sequential([
            self.normalizer,
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])


    def adapt(self, inputs):
        inputs = stack_dict(inputs)
        self.normalizer.adapt(inputs)


    def call(self, inputs):
        inputs = stack_dict(inputs)
        result = self.seq(inputs)
        return result

model = myModel()

model.adapt(dict(numeric_features))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)


numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
model.fit(numeric_dict_batches, epochs=5)

model.predict(dict(numeric_features.iloc[:3]))

#2- The Keras functional style

inputs = {}
for name, column in numeric_features.items():
    inputs[name] = tf.keras.Input(
        shape=(1,), name=name, dtype=tf.float32
    )
print(inputs)

x = stack_dict(inputs, fun=tf.concat)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

x = normalizer(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation ='relu')(x)
x = tf.keras.layers.Dense(1)(x)

model =tf.keras.Model(inputs, x)

model.compile(optimizer='adam',
              loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=True)

tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True)

model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)


numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
model.fit(numeric_dict_batches, epochs=20)
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






