import math

def cosine_sim(vec1, vec2):
	"""Let's convert the dictionaries to lists for easier matching."""
	vec1 = [val for val in vec1.values()]
	vec2 = [val for val in vec2.values()]

	dot_prod = 0
	for i, v in enumerate(vec1):
		dot_prod += v * vec2[i]

	mag1 = math.sqrt(sum([x**2 for x in vec1]))
	mag2 = math.sqrt(sum([x**2 for x in vec2]))

	return dot_prod / (mag1 * mag2)
