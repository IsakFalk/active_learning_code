"""
Example snakefile. This is only text, write your own needed functions.

from os.path import join

IDIR = '../include'
ODIR = 'obj'
LDIR = '../lib'

LIBS = '-lm'

CC = 'gcc'
CFLAGS = '-I' + IDIR


_HEADERS = ['hello.h']
HEADERS = [join(IDIR, hfile) for hfile in _HEADERS]

_OBJS = ['hello.o', 'hellofunc.o']
OBJS = [join(ODIR, ofile) for ofile in _OBJS]


rule hello:
    'build the executable from the object files'
    output:
        'hello'
    input:
        OBJS
    shell:
        "{CC} -o {output} {input} {CFLAGS} {LIBS}"

rule c_to_o:
    'compile a single .c file to an .o file'
    output:
        temp('{ODIR}/{name}.o')
    input:
        src='{name}.c',
        headers=HEADERS
    shell:
        "{CC} -c -o {output} {input.src} {CFLAGS}"

rule clean:
    'clean up temporary files'
    shell:
        "rm -f   *~  core  {IDIR}/*~"
"""

rule download_mnist:
    output:
        './data/external/mnist/'        
    shell:
        """
        mkdir -p {output}
        wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O {output}train-images-idx3-ubyte.gz
        wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O {output}train-labels-idx1-ubyte.gz
        wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O {output}t10k-images-idx3-ubyte.gz
        wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O {output}t10k-labels-idx1-ubyte.gz
        gunzip {output}*
        """
