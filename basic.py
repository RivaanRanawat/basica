from string_with_arrows import *
import string
# Constants

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

# Errors

class Error:
		def __init__(self, pos_start, pos_end, error_name, details):
				self.pos_start = pos_start
				self.pos_end = pos_end
				self.error_name = error_name
				self.details = details
		
		#displays arrows where the error came from
		def as_string(self):
				result  = f'{self.error_name}: {self.details}\n'
				result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
				result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
				return result

#When the operands data type dont match
class IllegalCharError(Error):
		def __init__(self, pos_start, pos_end, details):
				super().__init__(pos_start, pos_end, 'Illegal Character', details)

#When there is an invalid syntax
class InvalidSyntaxError(Error):
		def __init__(self, pos_start, pos_end, details=''):
				super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

#When there is divison by 0 etc
class RunTimeError(Error):
		def __init__(self, pos_start, pos_end, details, context):
				super().__init__(pos_start, pos_end, 'Run Time Error', details)
				self.context = context

class ExpectedCharError(Error):
		def __init__(self, pos_start, pos_end, details, context):
				super().__init__(pos_start, pos_end, 'Expected Character', details)
				self.context = context

		def as_string(self):
				result  = self.generate_traceback()
				result += f'{self.error_name}: {self.details}\n'
				result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
				return result
#because we wanna trace back
		def generate_traceback(self):
			result = ''
			pos = self.pos_start
			ctx = self.context

# while we have the parent the loop will go on
			while ctx:
				result = f' File {pos.fn}, line{str(pos.ln+1)}, in {ctx.display_name}\n' + result
				pos = ctx.parent_entry_pos
				ctx = ctx.parent

			return 'Traceback (most recent call last):\n' + result

# POSITION
#Checks the position and the column we are on
class Position:
		def __init__(self, idx, ln, col, fn, ftxt):
				self.idx = idx
				self.ln = ln
				self.col = col
				self.fn = fn
				self.ftxt = ftxt

		#moves ahead

		def advance(self, current_char=None):
				self.idx += 1
				self.col += 1

				if current_char == '\n':
						self.ln += 1
						self.col = 0

				return self

		def copy(self ):
				return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

# Tokens

TT_INT			= 'INT'
TT_FLOAT    = 'FLOAT'
TT_IDENTIFIER = 'IDENTIFIER'
TT_KEYWORD = 'KEYWORD'
TT_PLUS     = 'PLUS'
TT_MINUS    = 'MINUS'
TT_MUL      = 'MUL'
TT_DIV      = 'DIV'
TT_EQUALS   = 'EQUALS'
TT_LPAREN   = 'LPAREN'
TT_RPAREN   = 'RPAREN'
TT_EOF			= 'EOF'
TT_NOTEQUALTO = 'NOTEQUAL'
TT_LESSTHAN = 'LESSTHAN'
TT_GREATERTHAN = 'GREATERTHAN'
TT_LTE        = 'LTE'
TT_GTE        = 'GTE'
TT_POWER    = 'POWER'
TT_EQUIVALENT = 'EQUIVALENT'
TT_COMMA  = 'COMMA'
TT_ARROW = 'ARROW'

#Keywords

KEYWORDS = [
  'VAR',
  'AND',
  'OR',
  'NOT',
  'IF',
  'THEN',
  'ELIF',
  'ELSE',
  'FOR',
  'WHILE',
  'STEP',
  'TO',
  'FUN'
]

class Token:
		def __init__(self, type_, value=None, pos_start=None, pos_end=None):
				self.type = type_
				self.value = value

				if pos_start:
					self.pos_start = pos_start.copy()
					self.pos_end = pos_start.copy()
					self.pos_end.advance()

				if pos_end:
					self.pos_end = pos_end.copy()

#checks if the token matches with the given value and the type
		def matches(self, type_, value):
			return self.type == type_ and self.value == value
		
		 # if value is given, return data type of the value and value
		def __repr__(self):
				if self.value: return f'{self.type}:{self.value}'
				return f'{self.type}'

#Lexer

class Lexer:
		def __init__(self, fn, text):
				self.fn = fn
				self.text = text
				self.pos = Position(-1, 0, -1, fn, text)
				self.current_char = None
				self.advance()
		
		def advance(self):
				self.pos.advance(self.current_char)
				self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

#checks which operands and operators are there and then makes tokens
		def make_tokens(self):
				tokens = []

				while self.current_char != None:
						if self.current_char in ' \t':
								self.advance()
						elif self.current_char in DIGITS:
								tokens.append(self.make_number())
						elif self.current_char in LETTERS:
								tokens.append(self.make_identifier())
						elif self.current_char == '+':
								tokens.append(Token(TT_PLUS, pos_start=self.pos))
								self.advance()
						elif self.current_char == '-':
								tokens.append(self.make_minus_or_arrow())
						elif self.current_char == '*':
								tokens.append(Token(TT_MUL, pos_start=self.pos))
								self.advance()
						elif self.current_char == '/':
								tokens.append(Token(TT_DIV, pos_start=self.pos))
								self.advance()
						elif self.current_char == '^':
								tokens.append(Token(TT_POWER, pos_start=self.pos))
								self.advance()
						elif self.current_char == '(':
								tokens.append(Token(TT_LPAREN, pos_start=self.pos))
								self.advance()
						elif self.current_char == ')':
								tokens.append(Token(TT_RPAREN, pos_start=self.pos))
								self.advance()
						elif self.current_char == '!':
								tok, error = self.make_not_equals()
								if error: return [], error
								tokens.append(tok)
						elif self.current_char == '=':
								tokens.append(self.make_Equals())
						elif self.current_char == '<':
								tokens.append(self.make_less_than())
						elif self.current_char == '>':
								tokens.append(self.make_greater_than())
						elif self.current_char == ',':
								tokens.append(Token(TT_COMMA, pos_start=self.pos))
								self.advance()
						else:
								pos_start = self.pos.copy()
								char = self.current_char
								self.advance()
								return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

				tokens.append(Token(TT_EOF, pos_start=self.pos))
				return tokens, None

		#checks if the number is int or float
		def make_number(self):
				num_str = ''
				dot_count = 0
				pos_start = self.pos.copy()

				while self.current_char != None and self.current_char in DIGITS + '.':
						if self.current_char == '.':
								if dot_count == 1: break
								dot_count += 1
						num_str += self.current_char
						self.advance()

				if dot_count == 0:
						return Token(TT_INT, int(num_str), pos_start, self.pos)
				else:
						return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

		# makes and checks identifiers like variables

		def make_identifier(self):
		    id_str = ''
		    pos_start = self.pos.copy()

		    while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
			    id_str +=self.current_char
			    self.advance()

		#now we have to check if we have to build identifier token or keyword identifier
		    tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
		    return Token(tok_type, id_str, pos_start, self.pos)

		def make_minus_or_arrow(self):
			tok_type = TT_MINUS
			pos_start = self.pos.copy()
			self.advance()

			if self.current_char == '>':
				self.advance()
				tok_type = TT_ARROW

			return Token(tok_type, pos_start = pos_start, pos_end = self.pos)

		def make_not_equals(self):
			pos_start = self.pos.copy()
			self.advance()

			if self.current_char == "=":
				self.advance()
				return Token(TT_NOTEQUALTO, pos_start = pos_start, pos_end = self.pos), None

			self.advance()
			return None, ExpectedCharError(pos_start, self.pos, "Expected '=' after '!'")

		def make_Equals(self):
			tok_type = TT_EQUALS
			pos_start = self.pos.copy()
			self.advance()

			#determine if this going to be a = or ==
			if self.current_char == "=":
				self.advance()
				tok_type = TT_EQUIVALENT

			return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

		def make_less_than(self):
			tok_type = TT_LESSTHAN
			pos_start = self.pos.copy()
			self.advance()

			#determine if this going to be a <or <=
			if self.current_char == "=":
				self.advance()
				tok_type = TT_LTE

			return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

		def make_greater_than(self):
			tok_type = TT_GREATERTHAN
			pos_start = self.pos.copy()
			self.advance()

			#determine if this going to be a > or >=
			if self.current_char == "=":
				self.advance()
				tok_type = TT_GTE

			return Token(tok_type, pos_start=pos_start, pos_end=self.pos)
#NumberNode
#Takes the token that will be float or an int

class NumberNode:
	def __init__(self, tok):
		self.tok = tok

		self.pos_start = self.tok.pos_start
		self.pos_end = self.tok.pos_end

    #returns token in the string
	def __repr__(self):
		return f'{self.tok}'

class VarAccessNode:
	def __init__(self, var_name_tok):
		self.var_name_tok = var_name_tok

		self.pos_start = self.var_name_tok.pos_start
		self.pos_end = self.var_name_tok.pos_end
class VarAssignNode:
	def __init__(self, var_name_tok, value_node):
		self.var_name_tok = var_name_tok
		self.value_node = value_node

		self.pos_start = self.var_name_tok.pos_start
		self.pos_end = self.value_node.pos_end

#for add, subtract, multiply
class BinOpNode:
	def __init__(self, left_node, op_tok, right_node):
		self.left_node = left_node
		self.op_tok = op_tok
		self.right_node = right_node

		#Start position is in the start of the left node 

		self.pos_start = self.left_node.pos_start
		#End position is in the end of the right node
		self.pos_end = self.right_node.pos_end

	def __repr__(self):
		return f'({self.left_node}, {self.op_tok}, {self.right_node})'

#for right parenthesis, left parentheses and -n

class UnaryOpNode:
	def __init__(self, op_tok, node):
		self.op_tok = op_tok
		self.node = node
		self.pos_start = self.op_tok.pos_start
		self.pos_end = node.pos_end

	def __repr__(self):
		return f'({self.op_tok}, {self.node})'

class IfNode:
	def __init__(self, cases, else_case):
		self.cases = cases
		self.else_case = else_case

		self.pos_start = self.cases[0][0].pos_start
		self.pos_end = (self.else_case or self.cases[len(self.cases)-1][0]).pos_end

class ForNode:
	def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node):
		self.var_name_tok = var_name_tok
		self.start_value_node = start_value_node
		self.end_value_node = end_value_node
		self.step_value_node = step_value_node
		self.body_node = body_node

		self.pos_start = self.var_name_tok.pos_start
		self.pos_end = self.body_node.pos_end

class WhileNode:
	def __init__(self, condition_node, body_node):
		self.condition_node = condition_node
		self.body_node = body_node

		self.pos_start = self.condition_node.pos_start
		self.pos_end = self.body_node.pos_end


class FuncDefNode:
	# var_name_tok = none if the function is anonymous
	#arg_name_toks = arguments
	#body_node = evaluated when function is called(might have lists in the future because the function could be multiline)
	def __init__(self, var_name_tok, arg_name_toks, body_node):
		self.var_name_tok = var_name_tok
		self.arg_name_toks = arg_name_toks
		self.body_node = body_node

		#if the function has the name then pos_start is in functions name
		if self.var_name_tok:
			self.pos_start = self.var_name_tok.pos_start
		#if there is argument then go to the first argument
		elif len(self.arg_name_toks) > 0:
			self.pos_start = self.arg_name_toks[0].pos_start
		#else go to body node
		else:
			self.pos_start = self.body_node.pos_start

		self.pos_end = self.body_node.pos_end

class CallNode:
	def __init__(self, node_to_call, arg_nodes):
		self.node_to_call = node_to_call
		self.arg_nodes = arg_nodes

		self.pos_start = self.node_to_call.pos_start

		if len(self.arg_nodes) > 0:
			self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
		else:
			self.pos_end = self.node_to_call.pos_end
#Parser Result- check if there are errors
class ParseResult:
	def __init__(self):
		self.error = None
		self.node = None
		self.advance_count = 0

	def register_advancement(self):
		self.advance_count +=1

	def register(self, res):
		self.advance_count += res.advance_count
		if res.error: self.error = res.error
		return res.node

	def success(self, node):
		self.node = node
		return self

	def failure(self, error):
		if not self.error or self.advance_count == 0:
		    self.error = error
		return self

#Parser

class Parser:
	def __init__(self, tokens):
		self.tokens = tokens
		self.tok_idx = -1
		self.advance()

#grab current token
	def advance(self, ):
		self.tok_idx += 1
		if self.tok_idx < len(self.tokens):
			self.current_tok = self.tokens[self.tok_idx]
		return self.current_tok

	def parse(self):
		res = self.expr()
		if not res.error and self.current_tok.type != TT_EOF:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected '+', '-', '*', '/', '^', '==', '!=', '<', '>', <=', '>=', 'AND' or 'OR'"
			))
		return res

	#check grammar rules
	def term(self):
		return self.bin_op(self.factor, (TT_MUL, TT_DIV))

	def arith_expr(self):
		return self.bin_op(self.term, (TT_PLUS, TT_MINUS))


	def comp_expr(self):
		res = ParseResult()

		if self.current_tok.matches(TT_KEYWORD, 'NOT'):
			op_tok = self.current_tok
			res.register_advancement()
			self.advance()

			node = res.register(self.comp_expr())
			if res.error: return res
			return res.success(UnaryOpNode(op_tok, node))

		node = res.register(self.bin_op(self.arith_expr, (TT_EQUIVALENT, TT_NOTEQUALTO, TT_LESSTHAN, TT_GREATERTHAN, TT_LTE, TT_GTE)))

		if res.error:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
			    "Expected int, float, identifier, '+', '(' or 'NOT' "
		    ))

		return res.success(node)

	#check grammar rules

	def expr(self):
		res = ParseResult()


		if self.current_tok.matches(TT_KEYWORD, 'VAR'):
			res.register_advancement()
			self.advance()

			if self.current_tok.type != TT_IDENTIFIER:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected Identifier"
				))

			var_name = self.current_tok
			res.register_advancement()
			self.advance()

			if self.current_tok.type != TT_EQUALS:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected '='"
				))

			res.register_advancement()
			self.advance()

			expr = res.register(self.expr())
			if res.error: return res
			return res.success(VarAssignNode(var_name, expr))

		node =  res.register(self.bin_op(self.comp_expr, ((TT_KEYWORD, "AND"), (TT_KEYWORD, "OR"))))

		if res.error:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-' or '('"
			))

		return res.success(node)

	#check grammar rules

	def factor(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type in (TT_PLUS, TT_MINUS):
			res.register_advancement()
			self.advance()
			factor = res.register(self.factor())
			if res.error: return res
			return res.success(UnaryOpNode(tok, factor))

		return self.power()

	#for x^n 

	def power(self):
		return self.bin_op(self.call, (TT_POWER, ), self.factor)

	def call(self):
		res = ParseResult()
		atom = res.register(self.atom())
		if res.error: return res

		if self.current_tok.type == TT_LPAREN:
			res.register_advancement()
			self.advance()
			arg_nodes = []

			if self.current_tok.type == TT_RPAREN:
				res.register_advancement()
				self.advance()
			else:
				arg_nodes.append(res.register(self.expr()))
				if res.error:
					return res.failure(InvalidSyntaxError(
						self.current_tok.pos_start, self.current_tok.pos_end,
						"Expected ')', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(' or 'NOT'"
					))

				while self.current_tok.type == TT_COMMA:
					res.register_advancement()
					self.advance()

					arg_nodes.append(res.register(self.expr()))
					if res.error: return res

				if self.current_tok.type != TT_RPAREN:
					return res.failure(InvalidSyntaxError(
						self.current_tok.pos_start, self.current_tok.pos_end,
						f"Expected ',' or ')'"
					))

				res.register_advancement()
				self.advance()
			return res.success(CallNode(atom, arg_nodes))
		return res.success(atom)

	#+ and - included in the atom method because atom is only called in power and power is only called in factor. 

	def atom(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type in (TT_INT, TT_FLOAT):
			res.register_advancement()
			self.advance()
			return res.success(NumberNode(tok))

		#if it comes across an identifier, it will return the VarAccessNode
		elif tok.type == TT_IDENTIFIER:
			res.register_advancement()
			self.advance()
			return res.success(VarAccessNode(tok))

		elif tok.type == TT_LPAREN:
			res.register_advancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error: return res
			if self.current_tok.type == TT_RPAREN:
				res.register_advancement()
				self.advance()
				return res.success(expr)
			else:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected ')'"
				))

		elif tok.matches(TT_KEYWORD, 'IF'):
			if_expr = res.register(self.if_expr())
			if res.error: return res
			return res.success(if_expr)

		elif tok.matches(TT_KEYWORD, 'FOR'):
			for_expr = res.register(self.for_expr())
			if res.error: return res
			return res.success(for_expr)

		elif tok.matches(TT_KEYWORD, 'WHILE'):
			while_expr = res.register(self.while_expr())
			if res.error: return res
			return res.success(while_expr)

		elif tok.matches(TT_KEYWORD, 'FUN'):
			func_def = res.register(self.func_def())
			if res.error: return res
			return res.success(func_def)

		return res.failure(InvalidSyntaxError(
			tok.pos_start, tok.pos_end,
			"Expected int, float, identifier, 'IF', 'FOR', 'WHILE', 'FUN', '+' or '-'"
		))

	def if_expr(self):
		res = ParseResult()
		cases = []
		else_case = None

		if not self.current_tok.matches(TT_KEYWORD, 'IF'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'IF'"
			))

		res.register_advancement()
		self.advance()

		condition = res.register(self.expr())
		if res.error: return res

		if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'THEN'"
			))

		res.register_advancement()
		self.advance()

		expr = res.register(self.expr())
		if res.error: return res
		cases.append((condition, expr))

		while self.current_tok.matches(TT_KEYWORD, 'ELIF'):
			res.register_advancement()
			self.advance()

			condition = res.register(self.expr())
			if res.error: return res

			if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected 'THEN'"
				))

			res.register_advancement()
			self.advance()

			expr = res.register(self.expr())
			if res.error: return res
			cases.append((condition, expr))

		if self.current_tok.matches(TT_KEYWORD, 'ELSE'):
			res.register_advancement()
			self.advance()

			else_case = res.register(self.expr())
			if res.error: return res

		return res.success(IfNode(cases, else_case))

	def for_expr(self):
		res = ParseResult()

		if not self.current_tok.matches(TT_KEYWORD, 'FOR'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'FOR'"
			))

		res.register_advancement()
		self.advance()

		if self.current_tok.type != TT_IDENTIFIER:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected Identifier"
			))

		var_name = self.current_tok
		res.register_advancement()
		self.advance()

		if self.current_tok.type != TT_EQUALS:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected '='"
			))
		
		res.register_advancement()
		self.advance()

		start_value = res.register(self.expr())
		if res.error: return res

		if not self.current_tok.matches(TT_KEYWORD, 'TO'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'TO'"
			))
		
		res.register_advancement()
		self.advance()

		end_value = res.register(self.expr())
		if res.error: return res

		if self.current_tok.matches(TT_KEYWORD, 'STEP'):
			res.register_advancement()
			self.advance()

			step_value = res.register(self.expr())
			if res.error: return res
		else:
			step_value = None

		if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'THEN'"
			))

		res.register_advancement()
		self.advance()

		body = res.register(self.expr())
		if res.error: return res

		return res.success(ForNode(var_name, start_value, end_value, step_value, body))


	def while_expr(self):
		res = ParseResult()

		if not self.current_tok.matches(TT_KEYWORD, 'WHILE'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'WHILE'"
			))

		res.register_advancement()
		self.advance()

		condition = res.register(self.expr())
		if res.error: return res

		if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'THEN'"
			))

		res.register_advancement()
		self.advance()

		body = res.register(self.expr())
		if res.error: return res

		return res.success(WhileNode(condition, body))

	def func_def(self):
		res = ParseResult()

		if not self.current_tok.matches(TT_KEYWORD, 'FUN'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'FUN'"
			))

		res.register_advancement()
		self.advance()

		if self.current_tok.type == TT_IDENTIFIER:
			var_name_tok = self.current_tok
			res.register_advancement()
			self.advance()
			if self.current_tok.type != TT_LPAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected '('"
				))
		else:
			var_name_tok = None
			if self.current_tok.type != TT_LPAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected identifier or '('"
				))
		
		res.register_advancement()
		self.advance()
		arg_name_toks = []

		if self.current_tok.type == TT_IDENTIFIER:
			arg_name_toks.append(self.current_tok)
			res.register_advancement()
			self.advance()
			
			while self.current_tok.type == TT_COMMA:
				res.register_advancement()
				self.advance()

				if self.current_tok.type != TT_IDENTIFIER:
					return res.failure(InvalidSyntaxError(
						self.current_tok.pos_start, self.current_tok.pos_end,
						f"Expected identifier"
					))

				arg_name_toks.append(self.current_tok)
				res.register_advancement()
				self.advance()
			
			if self.current_tok.type != TT_RPAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected ',' or ')'"
				))
		else:
			if self.current_tok.type != TT_RPAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected identifier or ')'"
				))

		res.register_advancement()
		self.advance()

		if self.current_tok.type != TT_ARROW:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected '->'"
			))

		res.register_advancement()
		self.advance()
		node_to_return = res.register(self.expr())
		if res.error: return res

		return res.success(FuncDefNode(
			var_name_tok,
			arg_name_toks,
			node_to_return
		))


	def bin_op(self, func_a, ops, func_b=None):
		if func_b == None:
			func_b = func_a


		res = ParseResult()
		#left gets assigned to node and not the whole ParseResult
		left = res.register(func_a())
		if res.error: return res

		while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
			op_tok = self.current_tok
			res.register_advancement()
			self.advance()
			right = res.register(func_b())
			if res.error: return res
			left = BinOpNode(left, op_tok, right)

		return res.success(left)


#RunTime Result
class RunTimeResult:
	def __init__(self):
		self.value = None
		self.error = None

	def register(self, res):
		if res.error: self.error = res.error
		return res.value

	def success(self, value):
		self.value = value
		return self

	def failure(self, error):
		self.error = error
		return self
#Values
#This class is made so thjat there are not much repetitions in the Number class
class Value:
	def __init__(self):
		self.set_pos()
		self.set_context()

	def set_pos(self, pos_start=None, pos_end=None):
		self.pos_start = pos_start
		self.pos_end = pos_end
		return self

	def set_context(self, context=None):
		self.context = context
		return self

	def added_to(self, other):
		return None, self.illegal_operation(other)

	def subbed_by(self, other):
		return None, self.illegal_operation(other)

	def multed_by(self, other):
		return None, self.illegal_operation(other)

	def dived_by(self, other):
		return None, self.illegal_operation(other)

	def powed_by(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_eq(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_ne(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_lt(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_gt(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_lte(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_gte(self, other):
		return None, self.illegal_operation(other)

	def anded_by(self, other):
		return None, self.illegal_operation(other)

	def ored_by(self, other):
		return None, self.illegal_operation(other)

	def notted(self):
		return None, self.illegal_operation(other)

	def execute(self, args):
		return RTResult().failure(self.illegal_operation())

	def copy(self):
		raise Exception('No copy method defined')

	def is_true(self):
		return False

	def illegal_operation(self, other=None):
		if not other: other = self
		return RTError(
			self.pos_start, other.pos_end,
			'Illegal operation',
			self.context
		)


#class for storing number and then doing operations with other numbers
class Number(Value):
	def __init__(self, value):
		self.value = value

	def set_pos(self, pos_start=None, pos_end=None):
		self.pos_start = pos_start
		self.pos_end = pos_end
		return self

	def set_context(self, context=None):
		self.context = context
		return self
		
	def added_to(self, another):
		if isinstance(another, Number):
			return Number(self.value + another.value).set_context(self.context), None

	def sub_by(self, another):
		if isinstance(another, Number):
			return Number(self.value - another.value).set_context(self.context), None

	def mul_to(self, another):
		if isinstance(another, Number):
			return Number(self.value * another.value).set_context(self.context), None

	def div_by(self, another):
		if isinstance(another, Number):
			if another.value == 0:
				return None, RunTimeError(
					another.pos_start, another.pos_end,
					'Division by zero',
					self.context
				)
			return Number(self.value / another.value).set_context(self.context), None

	def pow_to(self, another):
		if isinstance(another, Number):
			return Number(self.value ** another.value).set_context(self.context), None

	#int will convert boolean into 0 or 1 ; 0-false; 1-true
	def get_comparison_eq(self, other):
		if isinstance(other, Number):
			return Number(int(self.value == other.value)).set_context(self.context), None

	def get_comparison_ne(self, other):
		if isinstance(other, Number):
			return Number(int(self.value != other.value)).set_context(self.context), None

	def get_comparison_lt(self, other):
		if isinstance(other, Number):
			return Number(int(self.value < other.value)).set_context(self.context), None

	def get_comparison_gt(self, other):
		if isinstance(other, Number):
			return Number(int(self.value > other.value)).set_context(self.context), None

	def get_comparison_lte(self, other):
		if isinstance(other, Number):
			return Number(int(self.value <= other.value)).set_context(self.context), None

	def get_comparison_gte(self, other):
		if isinstance(other, Number):
			return Number(int(self.value >= other.value)).set_context(self.context), None

	def andedby(self, other):
		if isinstance(other, Number):
			return Number(int(self.value and other.value)).set_context(self.context), None

	def oredby(self, other):
		if isinstance(other, Number):
			return Number(int(self.value or other.value)).set_context(self.context), None

	def notted(self):
		return Number(1 if self.value == 0 else 0).set_context(self.context), None

	def copy(self):
		copy = Number(self.value)
		copy.set_pos(self.pos_start, self.pos_end)
		copy.set_context(self.context)
		return copy

	def is_true(self):
		return self.value != 0

	def __repr__(self):
		return str(self.value)

class Function(Value):
	def __init__(self, name, body_node, arg_names):
		super().__init__()
		self.name = name or "<anonymous>"
		self.body_node = body_node
		self.arg_names = arg_names

	def execute(self, args):
		res = RunTimeResult()
		interpreter = Interpreter()
		new_context = Context(self.name, self.context, self.pos_start)
		new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)

		if len(args) > len(self.arg_names):
			return res.failure(RunTimeError(
				self.pos_start, self.pos_end,
				f"{len(args) - len(self.arg_names)} too many args passed into '{self.name}'",
				self.context
			))

		if len(args) < len(self.arg_names):
			return res.failure(RTError(
				self.pos_start, self.pos_end,
				f"{len(self.arg_names) - len(args)} few args passed into '{self.name}'",
				self.context
			))

		for i in range(len(args)):
			arg_name = self.arg_names[i]
			arg_value = args[i]
			arg_value.set_context(new_context)
			new_context.symbol_table.set(arg_name, arg_value)

		value = res.register(interpreter.visit(self.body_node, new_context))
		if res.error: return res
		return res.success(value)

	def copy(self):
		copy = Function(self.name, self.body_node, self.arg_names)
		copy.set_context(self.context)
		copy.set_pos(self.pos_start, self.pos_end)
		return copy

	def __repr__(self):
		return f"<function {self.name}>"

#Holds current context of the program
class Context:
	def __init__(self, display_name, parent = None, parent_entry_pos=None):
		self.display_name = display_name
		self.parent = parent
		self.parent_entry_pos = parent_entry_pos
		self.symbol_table = None

#Symbol TABLE- Keeps track of all the variable names and thier values
#Once function comes in there will be local variables and so they can be stored but there will be some variables that will be stored in the parent which can be used gloablly
class SymbolTable:
	def __init__(self, parent=None):
		self.symbols = {}
		self.parent = parent

	def get(self, name):
		value = self.symbols.get(name, None)
		#also checks if there is a global symbol table 
		if value == None and self.parent:
			return self.parent.get(name)
		return value

	def set(self, name, value):
		self.symbols[name] = value

	def remove(self, name):
		del self.symols[name]

#Interpreter
class Interpreter:
	#Visits all the child node
	def visit(self, node, context):
		#this will create a string depending on the type of the node, eg: NumberNode
		method_name = f'visit_{type(node).__name__}'
		method = getattr(self, method_name, self.no_visit_method)
		return method(node, context)

	def no_visit_method(self, node, context):
		raise Exception(f'No visit{type(node).__name__} method defined')

	def visit_NumberNode(self, node, context):
		#Cant be failure because there are no operations happening
		return RunTimeResult().success(
			Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
		)

	def visit_VarAccessNode(self, node, context):
		res = RunTimeResult()
		var_name = node.var_name_tok.value
		value = context.symbol_table.get(var_name)

		if not value:
			return res.failure(RunTimeError(
				node.pos_start, node.pos_end,
				f"'{var_name}' is not defined",
				context
			))
		#this line is there because when we create a run time error, it points to the initialisation and not the expression
		value = value.copy().set_pos(node.pos_start, node.pos_end)
		return res.success(value)

	def visit_VarAssignNode(self, node, context):
		res= RunTimeResult()
		var_name = node.var_name_tok.value
		value = res.register(self.visit(node.value_node, context))
		if res.error: return res

		context.symbol_table.set(var_name, value)
		return res.success(value)

	def visit_BinOpNode(self, node, context):
		res = RunTimeResult()
		left = res.register(self.visit(node.left_node, context))
		if res.error: return res
		right = res.register(self.visit(node.right_node, context))
		if res.error: return res

		if node.op_tok.type == TT_PLUS:
			result, error = left.added_to(right)
		elif node.op_tok.type == TT_MINUS:
			result,error = left.sub_by(right)
		elif node.op_tok.type == TT_MUL:
			result,error = left.mul_to(right)
		elif node.op_tok.type == TT_DIV:
			result,error = left.div_by(right)
		elif node.op_tok.type == TT_POWER:
			result,error = left.pow_to(right)
		elif node.op_tok.type == TT_EQUIVALENT:
			result,error = left.get_comparison_eq(right)
		elif node.op_tok.type == TT_NOTEQUALTO:
			result,error = left.get_comparison_ne(right)
		elif node.op_tok.type == TT_LESSTHAN:
			result,error = left.get_comparison_lt(right)
		elif node.op_tok.type == TT_GREATERTHAN:
			result,error = left.get_comparison_gt(right)
		elif node.op_tok.type == TT_LTE:
			result,error = left.get_comparison_lte(right)
		elif node.op_tok.type == TT_GTE:
			result,error = left.get_comparison_gte(right)
		elif node.op_tok.matches(TT_KEYWORD, 'AND'):
			result,error = left.andedby(right)
		elif node.op_tok.matches(TT_KEYWORD, 'OR'):
			result,error = left.oredby(right)

		if error:
			return res.failure(error)
		else:
		    return res.success(result.set_pos(node.pos_start, node.pos_end))

	def visit_UnaryOpNode(self, node, context):
		res = RunTimeResult()
		number = res.register(self.visit(node.node, context))

		if res.error: return res

		error = None

		if node.op_tok.type == TT_MINUS:
			number, error = number.mul_to(Number(-1))
		elif node.op_tok.matches(TT_KEYWORD, 'NOT'):
			number, error = number.notted()

		if error:
			return res.failure(error)
		else:
			 return res.success(number.set_pos(node.pos_start, node.pos_end))

	def visit_IfNode(self, node, context):
		res = RunTimeResult()

		for condition, expr in node.cases:
			condition_value = res.register(self.visit(condition, context))
			if res.error: return res

			if condition_value.is_true():
				expr_value = res.register(self.visit(expr, context))
				if res.error: return res
				return res.success(expr_value)

		if node.else_case:
			else_value = res.register(self.visit(node.else_case, context))
			if res.error: return res
			return res.success(else_value)

		return res.success(None)

	def visit_ForNode(self, node, context):
		res = RunTimeResult()

		start_value = res.register(self.visit(node.start_value_node, context))
		if res.error: return res

		end_value = res.register(self.visit(node.end_value_node, context))
		if res.error: return res

		if node.step_value_node:
			step_value = res.register(self.visit(node.step_value_node, context))
			if res.error: return res
		else:
			step_value = Number(1)

		i = start_value.value

		if step_value.value >= 0:
			condition = lambda: i < end_value.value
		else:
			condition = lambda: i > end_value.value
		
		while condition():
			context.symbol_table.set(node.var_name_tok.value, Number(i))
			i += step_value.value

			res.register(self.visit(node.body_node, context))
			if res.error: return res

		return res.success(None)

	def visit_WhileNode(self, node, context):
		res = RunTimeResult()

		while True:
			condition = res.register(self.visit(node.condition_node, context))
			if res.error: return res

			if not condition.is_true(): break

			res.register(self.visit(node.body_node, context))
			if res.error: return res

		return res.success(None)

	def visit_FuncDefNode(self, node, context):
		res = RunTimeResult()

		func_name = node.var_name_tok.value if node.var_name_tok else None
		body_node = node.body_node
		arg_names = [arg_name.value for arg_name in node.arg_name_toks]
		func_value = Function(func_name, body_node, arg_names).set_context(context).set_pos(node.pos_start, node.pos_end)
		
		if node.var_name_tok:
			context.symbol_table.set(func_name, func_value)

		return res.success(func_value)

	def visit_CallNode(self, node, context):
		res = RunTimeResult()
		args = []

		value_to_call = res.register(self.visit(node.node_to_call, context))
		if res.error: return res
		#it will show any errors where we call it which is the illegal operation error
		value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

		for arg_node in node.arg_nodes:
			args.append(res.register(self.visit(arg_node, context)))
			if res.error: return res

		return_value = res.register(value_to_call.execute(args))
		if res.error: return res
		return res.success(return_value)


# RUN

global_symbol_table = SymbolTable()
global_symbol_table.set("NULL", Number(0))
global_symbol_table.set("FALSE", Number(0))
global_symbol_table.set("TRUE", Number(1))

def run(fn, text):
		# Generate tokens
		lexer = Lexer(fn, text)
		tokens, error = lexer.make_tokens()
		if error: return None, error
		
		# Generate AST
		parser = Parser(tokens)
		ast = parser.parse()
		if ast.error: return None, ast.error

		#Run Program
		interpreter = Interpreter()
		context = Context('<program>')
		context.symbol_table = global_symbol_table
		#Will get passed down to the tree
		result = interpreter.visit(ast.node, context)


		return result.value , result.error