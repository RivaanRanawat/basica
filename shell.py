import basic
#created an infinite loop to get the raw form of the input in console.
while True:
	text= input("basica > ")
	#will remove white spaces; also called as trim in java
	if text.strip() == "": continue
	result, error = basic.run('<stdin>', text)

	if error: print(error.as_string())
	elif result: 
		if len(result.elements) == 1:
			print(repr(result.elements[0]))
		else:
			print(repr(result))

	#because we want to print the repr instead of print