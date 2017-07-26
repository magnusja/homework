"""

Some simple logging functionality, inspired by rllab's logging.
Assumes that each diagnostic gets logged each iteration

Call logz.configure_output_dir() to start logging to a 
tab-separated-values file (some_folder_name/log.txt)

To load the learning curves, you can do, for example

A = np.genfromtxt('/tmp/expt_1468984536/log.txt',delimiter='\t',dtype=None, names=True)
A['EpRewMean']

"""

import os.path as osp, shutil, time, atexit, os, subprocess

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class G:
    output_dir = None
    output_file = None
    first_row = True
    log_headers = []
    log_current_row = {}

def configure_output_dir(d=None):
    """
    Set output directory to d, or to /tmp/somerandomnumber if d is None
    """
    g = G()
    g.output_dir = d or "/tmp/experiments/%i"%int(time.time())
    assert not osp.exists(g.output_dir), "Log dir %s already exists! Delete it first or use a different dir"%g.output_dir
    os.makedirs(g.output_dir)
    g.output_file = open(osp.join(g.output_dir, "log.txt"), 'w')
    atexit.register(g.output_file.close)
    try:
        cmd = "cd %s && git diff > %s 2>/dev/null"%(osp.dirname(__file__), osp.join(g.output_dir, "a.diff"))
        subprocess.check_call(cmd, shell=True) # Save git diff to experiment directory
    except subprocess.CalledProcessError:
        print("configure_output_dir: not storing the git diff, probably because you're not in a git repo")
    print(colorize("Logging data to %s"%g.output_file.name, 'green', bold=True))

    return g

def log_tabular(g, key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    """
    if g.first_row:
        g.log_headers.append(key)
    else:
        assert key in g.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
    assert key not in g.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
    g.log_current_row[key] = val

def dump_tabular(g):
    """
    Write all of the diagnostics from the current iteration
    """
    vals = []
    print("-"*37)
    for key in g.log_headers:
        val = g.log_current_row.get(key, "")
        if hasattr(val, "__float__"): valstr = "%8.3g"%val
        else: valstr = val
        print("| %15s | %15s |"%(key, valstr))
        vals.append(val)
    print("-"*37)
    if g.output_file is not None:
        if g.first_row:
            g.output_file.write("\t".join(g.log_headers))
            g.output_file.write("\n")
        g.output_file.write("\t".join(map(str,vals)))
        g.output_file.write("\n")
        g.output_file.flush()
    g.log_current_row.clear()
    g.first_row=False
