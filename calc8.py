""" SPI - Simple Pascal Interpreter """
import math
###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis

INTEGER, CHAR, PLUS, MINUS, MUL, DIV, MODE, POW, EXP, SQRT, LOG, SIN, COS, TAN, CTG, ACOS, ASIN, ATAN, ACTG, LPAREN, RPAREN, EOF = (
    'INTEGER', 'CHAR', 'PLUS', 'MINUS', 'MUL', 'DIV', 'MODE', 'POW','EXP', 'SQRT', 'LOG', 'SIN', 'COS','TAN','CTG', 'ACOS', 'ASIN', 'ATAN', 'ACTG', '(', ')', 'EOF'
)

class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        """String representation of the class instance.

        Examples:
            Token(INTEGER, 3)
            Token(PLUS, '+')
            Token(MUL, '*')
        """
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()


class Lexer(object):
    def __init__(self, text):
        # client string input, e.g. "4 + 2 * 3 - 6 / 2"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Invalid character')

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def integer(self):
        """Return a (multidigit) integer consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def name(self):
        result = ''
        while self.current_char is not None and self.current_char.isalpha():
            result += self.current_char
            self.advance()
        return str.upper(result)

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return Token(INTEGER, self.integer())

            if self.current_char.isalpha():
                op = self.name()
                return Token(op, op.lower())


            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(DIV, '/')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            if self.current_char == '%':
                self.advance()
                return Token(MODE, '%')


            if self.current_char == '^':
                self.advance()
                return Token(POW, '^')

            self.error()

        return Token(EOF, None)


###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################

class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first token taken from the input
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def factor(self):
        """factor : (PLUS | MINUS) factor | INTEGER | LPAREN expr RPAREN"""
        token = self.current_token
        if token.type == PLUS:
            self.eat(PLUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == MINUS:
            self.eat(MINUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == INTEGER:
            self.eat(INTEGER)
            return Num(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        elif token.type == CHAR:
            self.eat (CHAR)
            node = UnaryOp(token, self.factor())
            return node

    def term(self):
        """term : factor ((MUL | DIV) factor)*"""
        node = self.factor()

        while self.current_token.type in (MUL, DIV, MODE, POW):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
            elif token.type == DIV:
                self.eat(DIV)
            elif token.type == MODE:
                self.eat(MODE)
            elif token.type == POW:
                self.eat(POW)
            node = BinOp(left=node, op=token, right=self.factor())

        while self.current_token.type in (EXP, SQRT, LOG, SIN, COS, TAN, CTG , ACOS, ASIN, ATAN, ACTG):
            token = self.current_token
            if token.type == EXP:
                self.eat(EXP)
            elif token.type == SQRT:
                self.eat(SQRT)
            elif token.type == LOG:
                self.eat(LOG)
            elif token.type == SIN:
                self.eat(SIN)
            elif token.type == COS:
                self.eat(COS)
            elif token.type == TAN:
                self.eat(TAN)
            elif token.type == CTG:
                self.eat(CTG)
            elif token.type == ACOS:
                self.eat(ACOS)
            elif token.type == ASIN:
                self.eat(ASIN)
            elif token.type == ATAN:
                self.eat(ATAN)
            elif token.type == ACTG:
                self.eat(ACTG)


            node = UnaryOp(token, self.factor())


        return node

    def expr(self):
        """
        expr   : term ((PLUS | MINUS) term)*
        term   : factor ((MUL | DIV) factor)*
        factor : (PLUS | MINUS) factor | INTEGER | LPAREN expr RPAREN
        """
        node = self.term()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def parse(self):
        node = self.expr()
        if self.current_token.type != EOF:
            self.error()
        return node


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################

class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


class Interpreter(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser

    def visit_BinOp(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_Num(self, node):
        return node.value

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == PLUS:
            return +self.visit(node.expr)
        elif op == MINUS:
            return -self.visit(node.expr)
        elif node.op.type == EXP:
            return math.exp(self.visit(node.expr))
        elif node.op.type == SQRT:
            return self.visit(node.expr) ** 0.5
        elif node.op.type == LOG:
            return math.log10(self.visit(node.expr))
        elif node.op.type == 'SIN':
            return math.sin(math.radians(self.visit(node.expr)))
        elif node.op.type == COS:
            return math.cos(math.radians(self.visit(node.expr)))
        elif node.op.type == TAN:
            return math.tan(math.radians(self.visit(node.expr)))
        elif node.op.type == CTG:
            return math.atan(math.radians(self.visit(node.expr)))
        elif node.op.type == ACOS:
            return math.acos(math.radians(self.visit(node.expr)))
        elif node.op.type == ASIN:
            return math.asin(math.radians(self.visit(node.expr)))
        elif node.op.type == ATAN:
            return math.asin(math.radians(self.visit(node.expr)))
        elif node.op.type == ACTG:
            return math.asin(math.radians(self.visit(node.expr)))
        else:
            return '2'

    def interpret(self):
        tree = self.parser.parse()
        if tree is None:
            return ''
        return self.visit(tree)


def main():
    while True:
        try:
            try:
                text = raw_input('spi> ')
            except NameError:  # Python3
                text = input('spi> ')
        except EOFError:
            break
        if not text:
            continue

        lexer = Lexer(text)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        result = interpreter.interpret()
        print(result)


if __name__ == '__main__':
    main()