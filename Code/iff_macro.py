from macropy.core.macros import Macros
import ast
from macropy.core import real_repr, unparse

macros = Macros()

config = {
    'lastcall': False
}

def set_config(nconf):
    for k in nconf:
        config[k] = nconf[k]



@macros.block
def mif(tree, args, expand_macros, **kw):
    try:
        exec("a="+args[0].s, config)
    except:
        print('Exception!')
        config['a'] = False
    #print(config['a'])
    if config['a']:
        expanded_tree = expand_macros(tree)
        config['lastcall'] = True
        return expanded_tree
    else:
        config['lastcall'] = False
        return [ast.Pass()]

@macros.block
def melse(tree, expand_macros, **kw):
    #print(not config['lastcall'])
    return [ast.Pass()] if config['lastcall'] else expand_macros(tree)

@macros.decorator
def printcode(tree, expand_macros, **kw):
    expanded_tree = expand_macros(tree)
    print(unparse(expanded_tree))
    return expanded_tree
