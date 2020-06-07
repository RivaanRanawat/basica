import basic
#created an infinite loop to get the raw form of the input in console.
while True:
	text= input("basica > ")
	print(text)
	result, error = basic.run('<stdin>', text)

	if error: print(error.as_string())
	elif result: print(repr(result))

	#because we want to print the repr instead of print