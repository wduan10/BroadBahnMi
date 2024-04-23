import re # python regular expression matching module

in_filename = 'CNN.py'
out_filename = 'CNN_script.py'
with open(in_filename, 'r') as f_orig:
    script = re.sub(r'# In\[.*\]:\n','', f_orig.read())
with open(out_filename,'w') as fh:
    fh.write(script)