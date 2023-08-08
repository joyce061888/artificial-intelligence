"""
Code to train a small neural network recipe classifier on the Yummly dataset. 
Author: Carolyn Anderson
Date: 2/27/2022
"""
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import glob
from tensorflow.keras import layers
from transformers import DistilBertTokenizer, TFDistilBertModel

tf.random.set_seed(0)

MAX_LENGTH = 12
EPOCHS = 5
BATCH_SIZE = 36
checkpoint_dir = "./regression-ckpt"

# These are the category labels for the top-6 categories. They correspond
# to the last two columns of the file yummly_data_top6_train.csv.
CATEGORIES = [
    "Main Dishes",  # 0
    "Desserts",  # 1
    "Beverages",  # 2
    "Soups",  # 3
    "Salads",  # 4
    "Condiments and Sauces",  # 5
]

def make_model(input_shape, num_classes):
    bert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    inputs = keras.Input(shape=input_shape,dtype=tf.int32)
    last_hidden_states = bert(inputs).last_hidden_state
    """
    INSERT MODEL CODE HERE
    """
    # Task 1
   # x = layers.Flatten()(last_hidden_states)
  #  outputs = layers.Dense(num_classes, activation = "softmax")(x)
      # Task 2
    layer1 = layers.Dense(input_shape)(last_hidden_states)
    layer2 = layers.Dense(input_shape)(last_hidden_states)
    layer3 = layers.Dense(input_shape)(last_hidden_states)
    x = layers.Flatten()(layer3)
    outputs = layers.Dense(num_classes, activation = "softmax")(x)
    return keras.Model(inputs, outputs)


def make_array(dataframe):
    toFloats = dataframe.to_numpy().astype(float)
    return toFloats


def get_predicted_label_from_predictions(predictions):
    predicted_label = max([(predictions[l], l) for l in range(len(predictions))])[1]
    return predicted_label


def sample_and_print_predictions(test_features,test_data,model):
    # Predict labels for some items from the test set:
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    sample_predictions = model.predict(x=test_features[:10])
    for i,p in enumerate(sample_predictions):
        predicted_label = get_predicted_label_from_predictions(sample_predictions[i])
        print(test_data["name"][i],"was predicted to be class",predicted_label,"and is actually class",test_data["category_number"][i],"("+test_data["category"][i]+")")
        print("Predictions:",' '.join([str(s) for s in p]))
    return

def prep_bert_data(data,max_length):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokens = [tokenizer(d, truncation=True, padding='max_length', max_length=max_length)['input_ids'] for d in data]
    for i in tokens:
        assert len(i) == max_length
    return np.array(tokens)

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(latest_checkpoint)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint,
            custom_objects={"TFDistilBertModel":TFDistilBertModel.from_pretrained("distilbert-base-uncased")})
    print("Creating a new model")
    model = make_model(input_shape=MAX_LENGTH, num_classes=6)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def train():
    train_data = pd.read_csv("Yummly/yummly_data_top6_train.csv")
    test_data = pd.read_csv("Yummly/yummly_data_top6_test.csv")

    train_features = prep_bert_data(train_data["name"], MAX_LENGTH)
    train_labels = make_array(train_data[["category_number"]])

    test_features = prep_bert_data(test_data["name"], MAX_LENGTH)
    test_labels = make_array(test_data[["category_number"]])

    print("Done prepping data!")

    model = make_or_restore_model() #Retrieve model from a checkpoint or make model

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir + "/ckpt-loss={loss:.2f}",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False)

    model.fit(
        x=train_features,
        y=train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[model_checkpoint_callback],
        validation_data=(test_features, test_labels),
    )

    # Predict labels for some items from the test set:
    sample_and_print_predictions(test_features,test_data,model)


def interactive():

    model = make_or_restore_model() #Retrieve model from a checkpoint or make model

    print("-" * 80)
    print("Completed loading model")

    while True:
        title = input("Enter a recipe title (blank to exit): ")
        if title == "":
            return
        # The title must be in an array, since the model expects a minibatch.
        input_data = prep_bert_data([title], MAX_LENGTH)
        prediction = model.predict(input_data)
        print(prediction[0])
        print(
            "Predicted category:",
            CATEGORIES[get_predicted_label_from_predictions(prediction[0])],
        )


def usage():
    """Prints usage information on the command-line."""
    print("Usage:")
    print("- python3 bert_model.py train")
    print("- python3 bert_model.py interactive")

def print_performance_by_class(test_data,test_predictions):
   acc_list = [[],[],[],[],[],[]]
   for i,p in enumerate(test_predictions):
       predicted_label = get_predicted_label_from_predictions(test_predictions[i])
       true_label = test_data["category_number"][i]
       acc = 1 if predicted_label == true_label else 0
       acc_list[true_label].append(acc)
   avg_acc = [sum(a)/len(a) for a in acc_list]
   print("Accuracy by Category:")
   for i,a in enumerate(avg_acc):
       print("Category",i,":",a)
   return

def main():

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if len(sys.argv) != 2:
        usage()
        return

    action = sys.argv[1]
    if action == "train":
        train()
    elif action == "interactive":
        interactive()
    else:
        usage()
        sys.exit(1)
        
   # test_predictions = model.predict(x=test_features)
    #print_performance_by_class(test_data,test_predictions)

main()