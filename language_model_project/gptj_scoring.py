# import query_gptj
# 
# # 3.1 - GPT-J
# 
# # Retrieve the most likely sequence of next tokens, up to length 5:
# print(query_gptj.completion_query("My favorite food is",5))
# # Retrieve the top 5 most likely tokens and their probabilities:
# print(query_gptj.token_query("My favorite food is",5))
# # Retrieve the average probability of the listed completions:
# query_gptj.word_query("My favorite food is","pickles;pizza;rocks;ice cream")

import sys
import math
import query_gptj


def score_condition(conditions):
    """
    Calculates evaluation metric for each sentence.
    """
 #   print(conditions) # prints the dictionary for one index (one sentence for 5 countries + 1 neural sentence);
    # ex. {'A': ['30', 'A', 'At California college, students have either watched or heard of movies such as', 'US'],
    # 'B': ['30', 'B', 'At Busan college, students have either watched or heard of movies such as', 'South Korea'],
    # 'C': ['30', 'C', 'At London college, students have either watched or heard of movies such as', 'UK'],
    # 'D': ['30', 'D', 'At Tecate college, students have either watched or heard of movies such as', 'Mexico'],
    # 'E': ['30', 'E', 'At Beijing college, students have either watched or heard of movies such as', 'China'],
    # 'F': ['30', 'F', 'At the city college, students have either watched or heard of movies such as', 'Neural']}
    reference = conditions['F']  # neural condition
    print("reference")
    print(reference)
    completions = reference[4:16]
    print("completions")
    print(completions)
    keys = [completions[i] for i in range(0,len(completions),2)]
    reference_probs = [float(completions[i]) for i in range(1,len(completions),2)]
    reference_dict = dict(zip(keys,reference_probs))
    diff_dict = {}
    for key in conditions.keys():
        if key != 'F':
            condition = conditions[key]
           # print(condition)  # prints the list of all tokens in one line of dataset;
            # ex. ['30', 'A', 'At California college, students have either watched or heard of movies such as', 'US']
            suggested = [condition[4:16][i] for i in range(0,len(completions),2)]
            probs = [float(condition[4:16][i]) for i in range(1,len(completions),2)]
            cdict = dict(zip(suggested,probs))
            print("cdict")
            print(cdict)
            rmlist = []
            for s in cdict.keys():
                if s not in keys:
                    rmlist.append(s)
                 #   print(cdict['OTHER'])
                    cdict['OTHER'] += cdict[s]  # after gpt generates words into results --> if word not in dictionary puts it into OTHER category?
                  #  print(s) # words in OTHER category
            for s in rmlist:
                cdict.pop(s)
            diffs = 0
            for s in cdict.keys():
                diffs += abs(reference_dict[s] - cdict[s])  # get the probability difference between neural frame sentence and country-specific
            diff_dict[key] = diffs
    return diff_dict


def score_all_by_condition(diff_list):
    """
    Calculates average score of the country-specific sentences. 
    """
    diff_dict = {'A':0,'B':0,'C':0,'D':0,'E':0}
    name_dict = {'A':'US','B':'South Korea','C':'UK','D':'Mexico','E':'China'}	
    for d in diff_list:
        for k in d.keys():
            diff_dict[k] += d[k]
    for k in diff_dict.keys():
        print(f"{name_dict[k]}: {diff_dict[k]/len(diff_list)}")	


def main():
    """
    Reads in a file (ex. tsv dataset file) from command line and scores
    each sentence based on evaluation metric.
    """
    infile = sys.argv[1]
    items = [s.strip().split('\t') for s in open(infile,'r').readlines()]
    conditions = {}
    # for dataset in tsv file score each sentence
    for item in items:
        if item[0] in conditions:
            conditions[item[0]][item[1]] = item
        else:
            conditions[item[0]] = {item[1]:item}
    score_dicts = [score_condition(conditions[c]) for c in conditions.keys()]
    score_all_by_condition(score_dicts)


# for testing in command line: python gptj_scoring.py tv_prompts.tsv 
main()