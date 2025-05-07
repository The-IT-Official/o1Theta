import re

# Token patterns
token_specification = [
    ('NUMBER', r'\d+'),    # Integer
    ('ASSIGN', r'='),      # Assignment
    ('IDENT', r'[A-Za-z]+'), # Identifier
    ('SEMI', r';'),        # Semicolon
    ('SKIP', r'[ \t\n]+'), # Skip over spaces, tabs, and newlines
    ('MISMATCH', r'.'),    # Any other character
]

# Lexer function
def lexer(code):
    line_num = 1
    line_start = 0
    tokens = []
    for mo in re.finditer('|'.join('(?P<%s>%s)' % pair for pair in token_specification), code):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'SKIP':
            continue
        elif kind == 'MISMATCHED':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
        else:
            tokens.append((kind, value))
        
    return tokens

code = "int x = 5;"
print(lexer(code))