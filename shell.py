import basic
#created an infinite loop to get the raw form of the input in console.
while True:
	text= input("basica > ")
	print(text)
	result, error = basic.run('<stdin>', text)

	if error: print(error.as_string())
	else: print(result)