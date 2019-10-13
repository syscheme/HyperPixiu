
"""
This is an extension of python3.6/csv.py
"""

import re
from _csv import Error, __version__, writer, reader, register_dialect, \
                 unregister_dialect, get_dialect, list_dialects, \
                 field_size_limit, \
                 QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONNUMERIC, QUOTE_NONE, \
                 __doc__
from _csv import Dialect as _Dialect

from collections import OrderedDict
from io import StringIO

__all__ = ["QUOTE_MINIMAL", "QUOTE_ALL", "QUOTE_NONNUMERIC", "QUOTE_NONE",
           "Error", "Dialect", "__doc__", "excel", "excel_tab",
           "field_size_limit", "reader", "writer",
           "register_dialect", "get_dialect", "list_dialects", "Sniffer",
           "unregister_dialect", "__version__", "DictReader", "DictWriter",
           "unix_dialect"]

########################################################################
class MCsvReader:
    def __init__(self, fieldsDict, dialect="excel"):
                # , fieldnames=None, restkey=None, restval=None,
                #  dialect="excel", *args, **kwds):
        '''
        
        '''
        self._fieldsDict = fieldsDict if fieldsDict else {}

        self._fieldnames = fieldnames   # list of keys for the dict
        self.restkey = restkey          # key to catch long rows
        self.restval = restval          # default value for short rows
        self.reader = reader(f, dialect)
        self.line_num = 0

    def __iter__(self):
        return self

    @property
    def rowtypes(self):
        return self._fieldsDict.keys()

    def fieldnames(self, rowtype):
        if rowtype in self.rowtypes:
            return self._fieldsDict[rowtype]
        return []

    def __next__(self):
        row = next(self.reader)
        self.line_num = self.reader.line_num

        # unlike the basic reader, we prefer not to return blanks,
        # because we will typically wind up with a dict full of None
        # values
        while row == [] :
            row = next(self.reader)
        
        tag = row[0]
        erase(row[0])
        if tag.endwith('#') : # this is a header line
            self._fieldsDict[tag[:-1]] = row
        elif tag.endwith['>'] : # this is a value line
            fieldnames = self._fieldsDict[tag[:-1]]
            d = OrderedDict(zip(fieldnames, row))
            lf = len(self.fieldnames)
            lr = len(row)
            if lf < lr:
                d[self.restkey] = row[lf:]
            elif lf > lr:
                for key in self.fieldnames[lr:]:
                    d[key] = self.restval
            return d # yield(d)
        else :
            pass # should print a warning

########################################################################
class MCsvWriter:
    def __init__(self, f, fieldsDict, dialect="excel"):
        if not fieldsDict:
            raise ValueError("fieldsDict must be specified")

        self._fieldsDict = fieldsDict
        self.restval = restval          # for writing short dicts
        self.extrasaction = extrasaction
        self.writer = writer(f, dialect, *args, **kwds)

    def writeheader(self):
        header = dict(zip(self.fieldnames, self.fieldnames))
        self.writerow(header)

    def _dict_to_list(self, rowdict):
        if self.extrasaction == "raise":
            wrong_fields = rowdict.keys() - self.fieldnames
            if wrong_fields:
                raise ValueError("dict contains fields not in fieldnames: "
                                 + ", ".join([repr(x) for x in wrong_fields]))
        return (rowdict.get(key, self.restval) for key in self.fieldnames)

    def writerow(self, rowdict):
        return self.writer.writerow(self._dict_to_list(rowdict))

    def writerows(self, rowdicts):
        return self.writer.writerows(map(self._dict_to_list, rowdicts))
