import re

def stem(phrase):
	return ' '.join([re.findall('^(.*ss|.*?)(s)?$', word)[0][0].strip("'")
			 for word in phrase.lower().split()])

print("Stemming plurals for 'houses' gives: ", end="")
print(stem('houses'))
print("Stemming plurals from 'Doctor House's Calls' gives: ", end="")
print(stem("Doctor House's calls"))
