"""
Code to extract features from the Yummly dataset. 
Author: Carolyn Anderson
Date: 2/27/2022
"""

import json

def get_idx(i):
	idx = str(i)
	while len(idx) < 5:
		idx = '0'+idx
	return idx

def get_sugar_fat_protein(n):
	protein = 0
	sugar = 0
	fat = 0
	for att in n:
		name = att['attribute']
		if name == 'SUGAR':
			sugar = att['value']
		elif name == 'FAT':
			fat = att['value']
		elif name == 'Adjusted Protein':
			protein = att['value']
	return sugar,fat,protein

def get_cuisines(c):
	asian_list = ['Chinese','Indian','Thai','Japanese','Vietnamese']
	mediterranean_list = ['Greek','French','Moroccan','Italian']
	if 'Cajun & Creole' in c:
		if 'Southern & Soul Food' in c:
			c.remove('Cajun & Creole')
		else:
			c[c.index('Cajun & Creole')] = 'Southern & Soul Food'
	if len(c) == 1:
		return c[0]
	else:
		if 'Asian' in c:
			for i in c:
				if i in asian_list:
					c.remove('Asian')
					break
		if 'Mediterranean' in c:
			for i in c:
				if i in mediterranean_list:
					c.remove('Mediterranean')
					break

		if len(c) > 1 and'American' in c:
			c.remove('American')
		if len(c) > 1 and 'Kid-Friendly' in c:
			c.remove('Kid-Friendly')
		if len(c) > 1 and 'Mexican' in c and 'Southwestern' in c:
			c.remove('Southwestern')
		if len(c) > 1 and 'Barbecue' in c and 'Southwestern' in c:
			c.remove('Southwestern')
	return '+'.join(c)

def get_numeric_cuisine(cuisine,cuisine_list):
	c1 = None
	c2 = None
	c3 = None
	cs = cuisine.split('+')
	for i,c in enumerate(cuisine_list):
		if c == cs[0]:
			c1 = i
		if len(cs) > 1:
			if c == cs[1]:
				c2 = i 
			if len(cs) > 2:
				if c == cs[2]:
					c3 = i 
	return c1,c2,c3

def get_category(category):
	if category == 'Cocktails':
		category = 'Beverages'
	return category

def get_numeric_category(category,category_list):
	cat_num = None
	for i,c in enumerate(category_list):
		if c == category:
			cat_num = i
	return cat_num

def write_to_file(feature_lists,header):
	head = "\t".join(header)
	print(head)
	for f in feature_lists:
		line = "\t".join([str(i) for i in f])
		print(line)

def main():

	feature_lists = []
	cuisine_list = ['Italian', 'Barbecue', 'French', 'American', 'Chinese', 'Kid-Friendly', 'Southwestern', 'Thai', 'Mexican', 'Indian', 'Southern & Soul Food', 'Japanese', 'Spanish', 'Cuban', 'Asian', 'Mediterranean', 'German', 'Hawaiian', 'Portuguese', 'Greek', 'English', 'Irish', 'Moroccan', 'Hungarian', 'Vietnamese', 'Swedish']
	category_list = ['Main Dishes','Desserts','Beverages','Soups','Salads','Condiments and Sauces','Side Dishes','Appetizers','Breads','Lunch and Snacks','Breakfast and Brunch','Afternoon Tea']
	for i in range(1,27639): #27639
		features = []
		idx = get_idx(i)
		features.append(idx)
		recipe = json.load(open('Yummly/recipes/meta'+idx+'.json','r'))
		features.append(recipe['name'])
		features.append(recipe['rating'])
		features.append(recipe['numberOfServings'])
		sugar,fat,protein = get_sugar_fat_protein(recipe['nutritionEstimates'])
		features.append(sugar)
		features.append(fat)
		features.append(protein)
		cuisine = get_cuisines(recipe['attributes']['cuisine'])
		ncuisine1, ncuisine2, ncuisine3 = get_numeric_cuisine(cuisine,cuisine_list)
		features.append(cuisine)
		features.append(ncuisine1)
		features.append(ncuisine2)
		features.append(ncuisine3)
		features.append(recipe['id'])
		features.append(recipe['totalTimeInSeconds'])
		category = get_category(recipe['attributes']['course'][0])
		features.append(category)
		ncategory = get_numeric_category(category,category_list)
		features.append(ncategory)
		
		feature_lists.append(features)

	header = ["file_id","name","rating","servings","sugar","fat","protein","cuisine","cuisine_number_1","cuisine_number_2","cuisine_number_3","id","time","category","category_number"]
	write_to_file(feature_lists,header)


main()