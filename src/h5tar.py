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

def tar_utf8(fn_h5tar, fn_members, createmode='a', baseNameAsKey=False) :

    createmode = createmode if 'a'==createmode else 'w'
    ret = []
    if isinstance(fn_members, str):
        fn_members = [fn_members]

    with h5py.File(fn_h5tar, createmode) as h5file:
        g = h5file.create_group(GNAME_TEXT_utf8) if not GNAME_TEXT_utf8 in h5file.keys() else h5file[GNAME_TEXT_utf8]
        g.attrs['desc']         = 'text member files via utf-8 encoding and bzip2 compression'
        for m in fn_members:
            try :
                filesize = os.path.getsize(m)
            except:
                continue

            print('tar_utf8() adding %s\tsize:%s' % (m, filesize))
            if '.bz2' == m[-4:]:
                with open(m, 'rb') as mf:
                    compressed = mf.read()
                m = m[:-4]
                filesize = '%sz' % filesize
            else :
                with open(m, 'r') as mf:
                    all_lines = mf.read()
                    compressed = bz2.compress(all_lines.encode('utf8'))

            npbytes = np.frombuffer(compressed, dtype=np.uint8)
            k = m
            if baseNameAsKey: k = os.path.basename(k)
            else:
                while len(k) >0 and ('/' == k[0] or '.' == k[0]):
                    k=k[1:]
                k = quote(k).replace('/','%2F')
                
            if k in g.keys():
                del g[k]
            sub = g.create_dataset(k, data=npbytes)
            sub.attrs['size'] = filesize
            sub.attrs['csize'] = len(compressed)
            ret.append(m)

    return ret

def untar_utf8(fn_h5tar, fn_members =None) :

    with h5py.File(fn_h5tar, 'r') as h5file:
        if not GNAME_TEXT_utf8 in h5file.keys() :
            print('no group[utf8_bz2] in %s', fn_h5tar)
            return

        g = h5file[GNAME_TEXT_utf8]
        for m in g.keys():
            ofn = unquote(m)
            while len(ofn) >0 and ('/' == ofn[0] or '.' == ofn[0]):
                ofn=ofn[1:]
            
            if len(ofn) <=0:
                continue
            
            print('untar_utf8() extracting %s' % ofn)
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

    ret = []

    with h5py.File(fn_h5tar, 'r') as h5file:
        if not GNAME_TEXT_utf8 in h5file.keys() :
            print('no group[utf8_bz2] in %s', fn_h5tar)
            return

        g = h5file[GNAME_TEXT_utf8]
        for i in g.keys():
            ret.append({'name':unquote(i), 'csize': g[i].attrs['csize'], 'size': g[i].attrs['size']})
        return ret

def read_utf8(fn_h5tar, fn_member) :
    with h5py.File(fn_h5tar, 'r') as h5file:
        if not GNAME_TEXT_utf8 in h5file.keys() :
            print('no group[utf8_bz2] in %s', fn_h5tar)
            return

        g = h5file[GNAME_TEXT_utf8]
        m = quote(fn_member).replace('/','%2F')
        if not m in g.keys():
            print('member[%s] not found' % m)
            return ''
        
        compressed = g[m].value.tobytes()
        all_lines = bz2.decompress(compressed).decode('utf8')
        return all_lines

    return ''

########################################################################
if __name__ == '__main__':

    # tar_utf8('abc.h5t', ['SZ399997_KL5m20200615.json.bz2'])

    if len(sys.argv) <3:
        print('%s {list|s|create|c|extract|x|add|a|show|s} <tarfilename> [textfile1 [textfile2 ...]]' % os.path.basename(sys.argv[0]))
        exit(0)

    cmd = sys.argv[1]
    fn_h5tar = sys.argv[2]
    fn_members= sys.argv[3:]

    if 'l' == cmd[0]:
        lst = list_utf8(fn_h5tar)
        print('members:\n\t%s' % '\n\t'.join([ '%s\t%s/%s' % (i['name'], i['csize'], i['size']) for i in lst]) )
    elif ('c' == cmd[0] or 'a' == cmd[0]) and len(fn_members) >0:
        tar_utf8(fn_h5tar, fn_members, cmd[0])
    elif 'x' in cmd:
        untar_utf8(fn_h5tar, fn_members)
    elif 's' == cmd[0] and len(fn_members) >0:
        print(read_utf8(fn_h5tar, fn_members[0]))

