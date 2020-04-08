from macropy.core.macros import Macros
import ast
from macropy.core import unparse
from macropy.core.walkers import Walker
from copy import deepcopy
import importlib

macros = Macros()


container = {'config': {}, 'first_load': True}

def set_config(nconf):
    container['config'] = nconf
    container['first_load'] = False
    from . import macroNeurons
    importlib.reload(macroNeurons)


@macros.decorator
def printcode(tree, expand_macros, **kw):
    expanded_tree = expand_macros(tree)
    print(unparse(expanded_tree))
    return expanded_tree


@Walker
def makeexecute(tree, ctx=None, stop=None, **kw):
    if type(tree) is ast.Attribute and tree.attr == 'config' and type(tree.value) is ast.Name and tree.value.id == 'self':
        stop()
        return ast.Name(ctx['name'], ast.Load())
    if type(tree) is ast.Name:
        stop()
        ctx['abort'] = True


@Walker
def transform(tree, ctx = None,**kw):
    if type(tree) is ast.If:
        nctx = {'name': ctx, 'abort': False}
        executable = ast.Expression(makeexecute.recurse(deepcopy(tree.test), ctx=nctx))
        if not nctx['abort']:
            ast.fix_missing_locations(executable)
            try:
                b = eval(compile(executable, 'inside_macro', 'eval'), container['config'])
                return tree.body if b else tree.orelse
            except:
                print('Exception!')
        #print(real_repr(tree.test))
        #print(real_repr(executable))


@macros.decorator
def preprocess(tree, **kw):
    #print(tree)
    if tree.name in container['config']:
        tree = transform.recurse(tree, ctx=tree.name) #, init_ctx=tree.name
        print(unparse(tree))
    return tree


