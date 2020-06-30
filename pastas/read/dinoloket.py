"""
This file contains the classes that can be used to import groundwater level
data from dinoloket.nl.

TODO: Get rid of filternummer en opmerking in self.series

"""
from itertools import takewhile
from io import StringIO
import csv
import os

import numpy as np
import pandas as pd

from ..timeseries import TimeSeries


def read_dino(fname, variable='stand', factor=0.01, settings='oseries', epsg=28992):
    """This method can be used to import files from Dinoloket that contain
     groundwater level measurements (https://www.dinoloket.nl/)

    Parameters
    ----------
    variable
    factor
    fname: str
        Filename and path to a Dino file.

    Returns
    -------
    ts: pastas.TimeSeries
        returns a Pastas TimeSeries object or a list of objects.

    """

    # Read the file
    is_river_gauge = os.path.basename(fname).lower().startswith('p')
    if is_river_gauge:
        dino = DinoPeilschaal.from_file(fname)
    else:
        dino = DinoGrondwaterstand.from_file(fname)
      
    # create timeseries object
    ts = TimeSeries(dino.get_ts_series(variable, factor),
        name=dino.get_ts_name(), 
        metadata=dino.get_ts_meta(factor, epsg).to_dict(),
        settings=settings,
        )    
    return ts
        

class DinoDataset:
    # class attributes to be filled in subclass
    meta_index_col = []
    series_index_col = []
    group_meta = []
    group_series = []
    meta_cols = {}
    meta_level_cols = []
    series_cols = {}
    series_level_cols = []

    def __init__(self, header, meta, series):
        self.header = header
        self.meta = meta
        self.series = series

    @staticmethod
    def skip_blanks(line, reader):
        while (line is None) or (len(line) == 0):
            try:
                line = next(reader)
            except StopIteration:
                return
        return line
    
    @staticmethod
    def read_header(line, reader, header):
        while (len(line) > 0) and not line[0].startswith('Locatie'):
            key = line[0].rstrip(':').strip()
            values = line[1:]
            header[key] = values
            line = next(reader)
        return line

    @staticmethod
    def try_get(series, variable):
        try:
            return (series
                .loc[:, variable]
                )
        except KeyError:
            raise ValueError(
                "variable {var:} is not in this dataset. Please use one of "
                "the following keys: {keys:}".format(
                var=variable,
                keys=series.columns.tolist(),
                ))

    @classmethod
    def read_meta(cls, meta_header, reader, delimiter):
        meta_header = [c for c in meta_header if c]
        not_empty = lambda row: any(bool(r) for r in row)
        meta_rows = takewhile(not_empty, reader)
        meta_f = StringIO(
            '\n'.join(delimiter.join(r) for r in meta_rows)
            )
        return pd.read_csv(meta_f,
            delimiter=delimiter,
            index_col=cls.meta_index_col,
            header=None,
            names=meta_header,
            parse_dates=True,
            dayfirst=True,
            usecols=meta_header,
            )

    @classmethod
    def read_series(cls, series_header, f, delimiter):
        series_header = [c for c in series_header if c]
        series = pd.read_csv(f,
            delimiter=delimiter,
            index_col=cls.series_index_col,
            header=None,
            names=series_header,
            parse_dates=True,
            dayfirst=True,
            usecols=series_header,
            )
        return series

    @classmethod
    def from_file(cls, fname, delimiter=','):
        with open(fname) as f:
            reader = csv.reader(f, delimiter=delimiter)
            line = next(reader)

            # read header
            header = {}
            line = cls.read_header(line, reader, header)

            # skip blanks
            line = cls.skip_blanks(line, reader)

            # read header (abbrevations)           
            line = cls.read_header(line, reader, header)

            # skip blanks
            meta_header = cls.skip_blanks(line, reader)       
                
            # read metadata
            if meta_header is not None:
                meta = cls.read_meta(meta_header, reader, delimiter)
            else:
                meta = None   

            # skip blanks
            try:
                series_header = cls.skip_blanks(None, reader)
            except StopIteration:
                series_header = None

            # read series
            if series_header is not None:
                series = cls.read_series(series_header, f, delimiter)
            else:
                series = None
            
            return cls(header, meta, series)
                    
    def get_ts_meta(self, factor, epsg=None):
        if self.meta is None:
            return {k: np.nan for k in self.meta_cols}
        meta = (self.meta
            .groupby(level=self.group_meta)
            .last()
            .rename(columns={v: k for k, v in self.meta_cols.items()})
            .loc[:, [k for k in self.meta_cols]]            
            .iloc[0, :] 
            )
        meta.loc[self.meta_level_cols] *= factor

        if epsg is not None:
            meta.loc['projection'] = 'epsg:{:d}'.format(epsg)

        return meta

    def get_ts_series(self, variable, factor):
        if self.series is None:
            return pd.Series()
        series = (self.series
            .groupby(level=self.group_series)
            .first()
            .rename(columns={v: k for k, v in self.series_cols.items()})
            )
        series = self.try_get(series, variable)
        series *= factor

        return series


class DinoGrondwaterstand(DinoDataset):
    meta_index_col = [0, 1]
    series_index_col = [0, 1, 2]
    group_meta = [0, 1]
    group_series = [2,]

    meta_cols = {
        'x': 'X-coordinaat',
        'y': 'Y-coordinaat',
        'meetpunt': 'Meetpunt (cm t.o.v. NAP)',
        'maaiveld': 'Maaiveld (cm t.o.v. NAP)',
        'bovenkant_filter': 'Bovenkant filter (cm t.o.v. NAP)',
        'onderkant_filter': 'Onderkant filter (cm t.o.v. NAP)',
        }
    meta_level_cols = [
        'meetpunt', 'maaiveld', 'bovenkant_filter', 'onderkant_filter'
        ]

    series_cols = {
        'stand': 'Stand (cm t.o.v. NAP)',
        }
    series_level_cols = [
        'stand',
        ]

    def get_ts_name(self):
        loc, filt = self.meta.index[0]
        return '{loc:}_{filt:}'.format(
            loc=loc,
            filt=filt,
            )
            
    def get_ts_meta(self, factor, epsg=None):
        meta = super().get_ts_meta(factor, epsg)
        meta.loc['z'] = (meta
            .loc[['bovenkant_filter', 'onderkant_filter']]
            .mean()
            )
        return meta


class DinoPeilschaal(DinoDataset):
    meta_index_col = [0,]
    series_index_col = [0, 1]
    group_meta = [0,]
    group_series = [1,]

    meta_cols = {
        'x': 'X-coordinaat',
        'y': 'Y-coordinaat',
        }

    series_cols = {
        'stand': 'Stand (cm t.o.v. NAP)',
        }
    series_level_cols = [
        'stand',
        ]

    def get_ts_name(self):
        loc = self.meta.index[0]
        return '{loc:}'.format(
            loc=loc,
            )
