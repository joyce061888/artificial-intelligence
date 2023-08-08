import query_gptj
import sys
import unicodedata


# 3.1 - GPT-J
# Retrieve the most likely sequence of next tokens, up to length 5:
#print(query_gptj.completion_query("I'm a sixty year old woman living in Tokyo. For breakfast, I like to eat",5))
#print()
# Retrieve the top 5 most likely tokens and their probabilities:
# returns a dictionary -> key = next likely token, value probability of that token
# print(query_gptj.token_query("I'm a sixty year old woman living in Tokyo. For breakfast, I like to eat",5))
#print()
# Retrieve the average probability of the listed completions:
#query_gptj.word_query("My favorite food is","pickles;pizza;rocks;ice cream")
    
def readFileAndCalculate(infile, outfile):
    """
    Reads in dataset file (tv_prompts.tsv) and calls query_gpt.token_query for each line in the file
    to calculate probabilities for next likely 5 words; calculates the evaluation metric for the dataset.
    """
    # read the file and split them into token based on tab
    items = [s.strip().split('\t') for s in open(infile,'r').readlines()]
    #print(items)
    # for each line of information for a sentence, run the gpt query to get probabilities
    probabilities = [] # list of dictionaries of the next likely 5 words for each sentence
    for item in items:
        sentence = item[2]
        #print(sentence)
        # run the gpt query to get probabilities of next 5 tokens that complete the sentences
        nextLikelyWordsDict = query_gptj.token_query(sentence, 5)

        probabilities.append(nextLikelyWordsDict)
    #print(probabilities)
    # write the probabilities of each sentence's 5 next likely words in a new file
    writeToFile(outfile, probabilities, items) 
    
def writeToFile(outfile, probabilities, items):
    """
    Helper function for readFileAndCalculate() that inputs probability data for each sentence
    into a new file; outputs results of model performance to a new file.
    """
    f = open(outfile, "w")
    i = 0  # index for dictionary of next 5 words for each sentence in probabilities (a list of dicts)
    lineToWrite = ""
    for item in items:
        # writing the info. of sentence taken in (ids, sentence, and country)
        lineToWrite += item[0] + "\t" + item[1] + "\t" + item[2] + "\t" + item[3]
        # write each word and its probability into the file
        
        # for every sentence's next 5 likely words and their probabilities
        dataDict = probabilities[i]
        i += 1
        probSums = 0
        for word in dataDict.keys(): # for each word of the 5 next likely words for each sentence
            # if the predicted word is a new line append '\n' to the file not an actual new line
            if ("\n" in word):
                word_to_write = "\\n"
            else:
                # Handling possible unicode characters: Convert word to utf-8
                word_to_write = unicodedata.normalize('NFKD', word).encode('ascii','replace').decode('utf-8')
            #print(item[2])
            #print(word_to_write)
            probSums += dataDict[word]
            lineToWrite += "\t" + word_to_write + "\t" + str(dataDict[word])
        # add OTHER category
        lineToWrite += "\t" + "OTHER" + "\t" + str(max((1 - probSums),0)) + "\t"
        lineToWrite += "\n" # new line for next sentence's data of probabilities

   # print(lineToWrite)
    f.write(lineToWrite)
    f.close()

def main():
    """
    Reads in the dataset and evaluates the model. 
    """
    infile = sys.argv[1]
    outfile = sys.argv[2]
    readFileAndCalculate(infile, outfile)

main()
#readFileAndCalculate("tv_prompts.tsv", "tv_prompts_results.tsv")
#python probe_task.py tv_prompts.tsv tv_prompts_results.tsv
#python gptj_scoring.py tv_prompts_results.tsv