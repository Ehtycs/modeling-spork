import inspect    
import pdb


class Breakpoint(Exception):
    pass

def debug():    
    """ A custom breakpoint kind of thing. This captures the local scope,
    around the call to debug(), inserts it to global scope and raises an 
    exception to drop you off to IPython kernel REPL as if you are in the 
    line where you called debug(). 

    Warning: This absolutely does mess up the global scope so restarting the kernel
    after calling this is advisable, depending on your coding style."""

    stack = inspect.stack()
    try: 
        env = stack[1][0].f_locals  
        
        # FISHY: search the globals of the main python file which is run
        # in ipython. I guess it's the first module in the stack
        for s in stack:
            if s.function == '<module>':
                break
            
        globs = s[0].f_globals   
        
        # shove the globals visible for the caller to the global ctx
        for k,v in stack[1][0].f_globals.items():
            globs[k] = v
            
        # shove the locals visible to the caller to the global ctx too
        for k, v in env.items():
            globs[k] = v     
            
    finally:
        del stack
    
    raise Breakpoint("Dropping to IPython.")

