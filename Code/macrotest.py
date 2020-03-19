from .iff_macro import macros, iff

with iff("abc == 2"):
    print('Test')

#somefunc(3)