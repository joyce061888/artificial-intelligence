"""
Code to train a small regression classifier on the Yummly recipe dataset. 
Author: Carolyn Anderson
Date: 2/27/2022
"""

import pandas as pd
import math
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import backend as K
from tensorflow.keras import layers
tf.keras.utils.set_random_seed(0)

def make_model(input_shape, num_classes):
    """
    INSERT MODEL CODE HERE
    """
    inputs = keras.Input(shape=input_shape) # input layer
  #  x = layers.Flatten()(inputs) # flatten images to a single dimension -> don't need to flatten images to vector since input is given as a vector
    # select sigmoid or softmax based on numbers of classes 
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    outputs = layers.Dense(units, activation = activation)(inputs)
    return keras.Model(inputs, outputs)

# get the average accuracy for each cateogry
# test_labels list of labels for each recipe
# list of categories
def print_performance_by_class(test_labels, predictions):
    points = [[],[],[],[],[],[]]
    print("Accuracy by Category:")
    
    for i,p in enumerate(predictions):
        maxPredict = get_predicted_label_from_predictions(predictions[i])
        if maxPredict == test_labels[i]:
            points[int(test_labels[i][0])].append(1)
        else:
            points[int(test_labels[i][0])].append(0)
    
    result = []
    # iterate through each list in points, add all num in sublist, divide by its len
    # then put into result list then print each one out
    for l in points:
        if (len(l) == 0):
            result.append(0.0)
        result.append(round((sum(l)/len(l)), 3))
    
    for r in range(len(result)):
        print("Category " + str(r) + ": " + str(result[r]))


def add_strawberry_column(dataframe):
    names = dataframe["name"]
    has_strawberry = []
    for name in names:
        if "strawberry" in name.lower():
            has_strawberry.append(1) 
        else:
            has_strawberry.append(0)
    dataframe["strawberry"] = has_strawberry
    return dataframe;

def add_pea_column(dataframe):
    names = dataframe["name"]
    has_pea = []
    for name in names:
        if "pea" in name.lower():
            has_pea.append(1)
        else:
            has_pea.append(0)
    dataframe["pea"] = has_pea
    return dataframe;

def add_salad_column(dataframe):
    names = dataframe["name"]
    has_salad = []
    for name in names:
        if "salad" in name.lower():
            has_salad.append(15)
        else:
            has_salad.append(0)
    dataframe["salad"] = has_salad
    return dataframe;

def add_soups_column(dataframe):
    names = dataframe["name"]
    has_soups = []
    for name in names:
        if "soups" in name.lower():
            has_soups.append(15)
        else:
            has_soups.append(0)
    dataframe["soups"] = has_soups
    return dataframe;

def add_sauces_column(dataframe):
    names = dataframe["name"]
    has_sauces = []
    for name in names:
        if "sauces" in name.lower():
            has_sauces.append(15)
        else:
            has_sauces.append(0)
    dataframe["sauces"] = has_sauces
    return dataframe;

def make_array(dataframe):
	noNones = dataframe.applymap(lambda x:-1 if x == 'None' else x)
	toFloats = noNones.to_numpy().astype(float)
	return toFloats

def descriptive_stats(dataframe):
	dataframe = dataframe.applymap(lambda x:-1 if x == 'None' else x)
	for column_name in dataframe.columns:
		column = dataframe[column_name]
		try:
			column = column.astype(float)
			print(column_name.upper()," "*(18-len(column_name.upper())),"| Stdev:",round(column.std(),2),"\t","Mean:",round(column.mean(),2),"\t","Median:",round(column.median(),2))
		except:
			pass

def stats_by_label(dataframe):
	print(type(dataframe['category_number'][0]))
	for i in range(6):
		label_subset = dataframe.loc[dataframe['category_number'] == i]
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		print("DESCRIPTIVE STATS FOR CATEGORY",i)
		descriptive_stats(label_subset)
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def print_coefficients(model,selected_features):
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print("Model weights (rows=features; columns=categories)")
	print("Feature"+" "*13+"\t0(Mains)\t1(Desserts)\t2(Beverages)\t3(Soups)\t4(Salads)\t5(Sauces)")
	weights = model.layers[1].weights[0]
	for i,f in enumerate(weights):
		ws = [float(v) for v in f]
		print(selected_features[i]+" "*(20-len(selected_features[i]))+"\t"+'\t\t'.join(map(lambda x:str(round(x,3)),ws)))

def get_predicted_label_from_predictions(predictions):
	predicted_label = max([(predictions[l],l) for l in range(len(predictions))])[1]
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

def main():
	train_data = pd.read_csv('Yummly/yummly_data_top6_train.csv')
	test_data = pd.read_csv('Yummly/yummly_data_top6_test.csv')
	
	train_data = add_strawberry_column(train_data)
	test_data = add_strawberry_column(test_data)

	train_data = add_pea_column(train_data)
	test_data = add_pea_column(test_data)

	train_data = add_salad_column(train_data)
	test_data = add_salad_column(test_data)
	
	train_data = add_soups_column(train_data)
	test_data = add_soups_column(test_data)
	
	train_data = add_sauces_column(train_data)
	test_data = add_sauces_column(test_data)

	print(train_data.head())
	print(len(train_data))

	selected_features = ["rating", "time", "servings", "sugar", "fat", "protein", "cuisine_number_1", "cuisine_number_2", "cuisine_number_3", "strawberry", "pea", "salad", "soups", "sauces"]

	train_features = make_array(train_data[selected_features])
	train_labels = make_array(train_data[["category_number"]])
	test_features = make_array(test_data[selected_features])
	test_labels = make_array(test_data[["category_number"]])
	
	
    
	epochs = 5
	batch_size = 1

	model = make_model(input_shape=len(selected_features),num_classes=6)

	#Create an image of your network. If you can't install pydot, comment out the line below:
#	keras.utils.plot_model(model, show_shapes=True) 

	callbacks = []#[keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),] #uncomment to checkpoint
	model.compile(
    	optimizer=keras.optimizers.Adam(1e-4),
    	loss=tf.losses.SparseCategoricalCrossentropy(),
    	metrics=["accuracy"],
	)
	model.fit(
    	x=train_features,y=train_labels, epochs=epochs, \
    	batch_size=batch_size, callbacks=callbacks, validation_data=(test_features,test_labels),
	)

	# Predict labels for some items from the test set:
	sample_and_print_predictions(test_features,test_data,model)

	# Which features are most informative for each class?
	print_coefficients(model,selected_features)

	# Descriptive stats for features:
	stats_by_label(train_data)
	
	print_performance_by_class(test_labels, model.predict(x=test_features))

main()