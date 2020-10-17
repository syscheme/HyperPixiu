# encoding: UTF-8

'''
BackTest inherits from Trader
'''
from __future__ import division

import os, sys
import h5py, bz2
import numpy as np
from urllib.parse import quote, unquote

GNAME_TEXT_utf8 = 'utf8_bz2'

def tar_utf8(fn_h5tar, fn_members, createmode='a') :

    with h5py.File(fn_h5tar, createmode) as h5file:
        g = h5file.create_group(GNAME_TEXT_utf8) if not GNAME_TEXT_utf8 in h5file.keys() else h5file[GNAME_TEXT_utf8]
        g.attrs['desc']         = 'text member files via utf-8 encoding and bzip2 compression'
        for m in fn_members:
            try :
                filesize = os.path.getsize(m)
            except:
                continue

            k = quote(m).replace('/','%2F')
            if k in g.keys():
                del g[k]
                
            print('adding %s\tsize:%s' % (m, filesize))
            with open(m, 'r') as mf:
                all_lines = mf.read()
                compressed = bz2.compress(all_lines.encode('utf8'))
                npbytes = np.frombuffer(compressed, dtype=np.uint8)
                sub = g.create_dataset(k, data=npbytes)
                sub.attrs['size'] = filesize

def untar_utf8(fn_h5tar, fn_members =None) :

    with h5py.File(fn_h5tar, 'r') as h5file:
        if not GNAME_TEXT_utf8 in h5file.keys() :
            print('no group[utf8_bz2] in %s', fn_h5tar)
            return

        g = h5file[GNAME_TEXT_utf8]
        for m in g.keys():
            ofn = unquote(m)
            while len(ofn) >0 and '/' == ofn[0]:
                ofn=ofn[1:]
            
            if len(ofn) <=0:
                continue
            
            print('extracting %s' % ofn)
            dir = os.path.dirname(ofn)
            if len(dir) >0 and dir != ofn:
                try :
                    os.makedirs(dir)
                except FileExistsError:  pass
            
            compressed = g[m].value.tobytes()
            all_lines = bz2.decompress(compressed).decode('utf8')

            with open(ofn, 'w') as mf:
                mf.write(all_lines)

def list_utf8(fn_h5tar, fn_members =None) :

    with h5py.File(fn_h5tar, 'r') as h5file:
        if not GNAME_TEXT_utf8 in h5file.keys() :
            print('no group[utf8_bz2] in %s', fn_h5tar)
            return

        g = h5file[GNAME_TEXT_utf8]
        print('members:\n\t%s' % '\n\t'.join([ '%s\t%s' % (unquote(i), g[i].attrs['size']) for i in g.keys()]))

########################################################################
if __name__ == '__main__':

    # tar_utf8('abc.h5', ['/tmp/SinaWeek.20200615/SinaKL5m_20200615/SZ399997_KL5m20200615.json'])

    if len(sys.argv) <3:
        print('%s {list|s|create|c|extract|x|add|a} <tarfilename> [textfile1 [textfile2 ...]]' % os.path.basename(sys.argv[0]))
        exit(0)

    cmd = sys.argv[1]
    fn_h5tar = sys.argv[2]
    fn_members= sys.argv[3:]

    if 'l' == cmd[0]:
        list_utf8(fn_h5tar)
    elif ('c' == cmd[0] or 'a' == cmd[0]) and len(fn_members) >0:
        tar_utf8(fn_h5tar, fn_members, cmd[0])
    elif 'x' in cmd:
        untar_utf8(fn_h5tar, fn_members)

