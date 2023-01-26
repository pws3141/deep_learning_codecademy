import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# load the data
dataset = pd.read_csv("life_expectancy.csv")

# examine the data
dataset.head()
dataset.columns
dataset.describe()
dataset_shape = dataset.shape

# drop 'Country' column
# do not require to create a predictive model of life expectancy
dataset = dataset.drop(labels = "Country", axis = 1)

## split data into features and labels
# labels if 'Life expectancy'
labels = dataset.iloc[:, -1]
# check to see if correct
labels.shape
labels.head()

# features: all other columns
features = dataset.iloc[:, range(dataset_shape[1] - 1)]
# check to see if correct
features.shape
features.head()

# apply one-hot encoding to categorical data
features = pd.get_dummies(features)
features.shape
features.head()

# create training and test set
(features_train, features_test, 
labels_train, labels_test) = train_test_split(features, labels, train_size = 0.8,
                                             random_state = 1)
print(features_train.shape)
print(features_test.shape)
print(labels_train.shape)
print(labels_test.shape)

# standardise the data
# only standardise the non-categorical data
# i.e. ignore the one-hot encoded data
features_numerical = features.select_dtypes(include = ["float64", "int64"])
ct = ColumnTransformer([('standardize', StandardScaler(), features_numerical.columns)],
                            remainder = 'passthrough')

features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)


### Create keras Sequential model
# tensorflow.keras.models.Sequential
my_model = Sequential(name = "life_model")
input = InputLayer(input_shape = features.shape[1])
my_model.add(input)
# need one output from regression model
my_model.add(Dense(1))
# look at model summary
my_model.summary()

# using Adam opt
opt = Adam(learning_rate=0.01)
my_model.compile(loss='mse',  metrics=['mae'], optimizer=opt)

# train model
my_model.fit(features_train, labels_train, epochs = 40, batch_size = 1,
             verbose = 1)

#evaluate the model on the test data
val_mse, val_mae = my_model.evaluate(features_test, labels_test, verbose = 1)

print("MAE: ", val_mae)

#### Model Tuning

## learning rates
# create function that builds model

def fit_model(f_train, l_train, learning_rate, num_epochs, bs):
    #build the model
    model = Sequential(f_train, learning_rate)
    #train the model on the training data
    history = model.fit(f_train, l_train, epochs = num_epochs, batch_size = bs,
                        verbose = 0, validation_split = 0.2)
    # plot learning curves
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('lrate=' + str(learning_rate))
    plt.legend(loc="upper right")

#make a list of learning rates to try out
learning_rates = [1E-1, 1E-3, 1E-7]
#fixed number of epochs
num_epochs = 100
#fixed number of batches
batch_size = 10 

%matplotlib

for i in range(len(learning_rates)):
  plot_no = 420 + (i+1)
  plt.subplot(plot_no)
  fit_model(features_train, labels_train, learning_rates[i], num_epochs, batch_size)

plt.tight_layout()
plt.show()
#plt.savefig('static/images/my_plot.png')
print("See the plot on the right with learning rates", learning_rates)

