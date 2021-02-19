# #! /usr/bin/env python
# encoding: UTF-8

'''
BackTest inherits from Trader
'''
from __future__ import division

import os, sys
import h5py, bz2
import numpy as np
from urllib.parse import quote, unquote
import shutil

GNAME_TEXT_utf8 = 'utf8_bz2'

########################################################################
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

            print('tar_utf8() %s adding %s\tsize:%s' % (fn_h5tar, m, filesize))
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

# ----------------------------------------------------------------------
def untar_utf8(fn_h5tar, fn_members =None) :

    with h5py.File(fn_h5tar, 'r') as h5file:
        if not GNAME_TEXT_utf8 in h5file.keys() :
            print('no group[utf8_bz2] in %s' % fn_h5tar)
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
            
            compressed =g[m][()].tobytes() # compressed = g[m].value.tobytes()
            all_lines = bz2.decompress(compressed).decode('utf8')

            with open(ofn, 'w') as mf:
                mf.write(all_lines)

# ----------------------------------------------------------------------
def list_utf8(fn_h5tar, fn_members =None) :

    ret = []

    with h5py.File(fn_h5tar, 'r') as h5file:
        if not GNAME_TEXT_utf8 in h5file.keys() :
            print('no group[utf8_bz2] in %s' % fn_h5tar)
            return

        g = h5file[GNAME_TEXT_utf8]
        for i in g.keys():
            ret.append({'name':unquote(i), 'csize': g[i].attrs['csize'], 'size': g[i].attrs['size']})
        return ret

# ----------------------------------------------------------------------
def read_utf8(fn_h5tar, fn_member) :
    with h5py.File(fn_h5tar, 'r') as h5file:
        if not GNAME_TEXT_utf8 in h5file.keys() :
            print('no group[utf8_bz2] in %s', fn_h5tar)
            return

        g = h5file[GNAME_TEXT_utf8]
        m = quote(fn_member).replace('/','%2F')
        if not m in g.keys():
            print('member[%s] not found in %s' % (m, fn_h5tar))
            return ''

        compressed =g[m][()].tobytes() # compressed = g[m].value.tobytes()
        all_lines = bz2.decompress(compressed).decode('utf8')
        return all_lines

    return ''

# ----------------------------------------------------------------------
def write_utf8(fn_h5tar, memberName, text, createmode='a') :

    createmode = createmode if 'a'==createmode else 'w'
    with h5py.File(fn_h5tar, createmode) as h5file:
        g = h5file.create_group(GNAME_TEXT_utf8) if not GNAME_TEXT_utf8 in h5file.keys() else h5file[GNAME_TEXT_utf8]
        g.attrs['desc']         = 'text member files via utf-8 encoding and bzip2 compression'

        tsize = len(text)
        print('tar_utf8() %s adding %s  text-size:%s' % (fn_h5tar, memberName, tsize))
        compressed = bz2.compress(text.encode('utf8'))
        zsize = len(compressed)
        npbytes = np.frombuffer(compressed, dtype=np.uint8)
        k = memberName
        while len(k) >0 and ('/' == k[0] or '.' == k[0]):
            k=k[1:]
        k = quote(k).replace('/','%2F')
            
        if k in g.keys():
            del g[k]

        sub = g.create_dataset(k, data=npbytes)
        sub.attrs['size'] = tsize
        sub.attrs['csize'] = len(compressed)
        return zsize, tsize

# ----------------------------------------------------------------------
def h5shrink(fn_h5) :
    ogns, gns = [], []
    with h5py.File(fn_h5, 'r') as h5in:
        with h5py.File(fn_h5 + "~", 'w') as h5out:
            ogns = list(h5in.keys()) 
            for gn in ogns:
                g = h5in[gn]
                h5in.copy(g.name, h5out) # note the destGroup is the parent where the group want to copy under-to
                # go = h5out[gn]
                gns.append(gn)
    
    print('%s refreshed %d groups' % (fn_h5, len(gns)))
    # try:
    if len(gns) >0 and len(gns) == len(ogns) :
        os.remove(fn_h5)
        shutil.move(fn_h5 + "~", fn_h5)
    # except: pass

# ----------------------------------------------------------------------
# https://www.cnblogs.com/osnosn/p/12574976.html
def h5visit(f, tab=''):
    print(tab,'Group:',f.name,'len:%d'%len(f))
    mysp2=tab[:-1]+ '  |-*'
    for vv in f.attrs.keys():  # 打印属性
        print(mysp2, end=' ')
        print('%s = %s'% (vv,f.attrs[vv]))

    mysp=tab[:-1] + '  |-'
    for k in f.keys():
        d = f[k]
        if isinstance(d,h5py.Group):
            h5visit(d,mysp)
            continue

        if not isinstance(d,h5py.Dataset):
            print('??->',d,'Unkown Object!')
            continue

        print(mysp, 'Dataset:', d.name, '%s[%d]'%(d.dtype, d.size))
        mysp1=mysp[:-1]+ '  |-'
        if d.dtype.names is not None:
            print(mysp,end=' ')
            for vv in d.dtype.names:
                print(vv,end=',')
            print()

        mysp2=mysp1[:-1]+ '  |-*'
        for vv in d.attrs.keys():  # 打印属性
            print(mysp2,end=' ')
            try:
                print('%s = %s'% (vv, d.attrs[vv]))
            except TypeError as e:
                print('%s = %s'% (vv,e))
            except:
                print('%s = ?? Other ERR'% (vv,))
        
        #print(d[:12])  # 打印12组数据看看

########################################################################
if __name__ == '__main__':
    # h5shrink('/tmp/SinaKL5m_20210210-2.h5t')
    # tar_utf8('abc.h5t', ['SZ399997_KL5m20200615.json.bz2'])

    if len(sys.argv) <3:
        print('%s {list|l|create|c|extract|x|add|a|show|s|visit|v|shrink|k} <tarfilename> [textfile1 [textfile2 ...]]' % os.path.basename(sys.argv[0]))
        exit(0)

    cmd = sys.argv[1]
    fn_h5tar = sys.argv[2]
    fn_members= sys.argv[3:]

    if cmd in ['list', 'l']:
        lst = list_utf8(fn_h5tar)
        print('members:\n\t%s' % '\n\t'.join([ '%s\t%s/%s' % (i['name'], i['csize'], i['size']) for i in lst]) )
    elif cmd in ['create', 'c', 'add', 'a'] and len(fn_members) >0:
        tar_utf8(fn_h5tar, fn_members, cmd[0])
    elif cmd in ['extract', 'x']:
        untar_utf8(fn_h5tar, fn_members)
    elif cmd in ['show', 's'] and len(fn_members) >0:
        print(read_utf8(fn_h5tar, fn_members[0]))
    elif cmd in ['visit', 'v'] and len(fn_h5tar) >0:
        with h5py.File(fn_h5tar, 'r') as f:
            h5visit(f)
    elif cmd in ['shrink', 'k'] and len(fn_h5tar) >0:
        h5shrink(fn_h5tar)
    else:
        print('%s illegal subcommand %s' % (os.path.basename(sys.argv[0]), cmd))

