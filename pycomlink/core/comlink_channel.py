# ----------------------------------------------------------------------------
# Name:         comlink_channel
# Purpose:      Class that represents one channel of a CML, holding the
#               TX and RX data as well as info on frequency
#               and polarization.
#
# Authors:      Christian Chwala
#
# Created:      24.02.2016
# Copyright:    (c) Christian Chwala 2016
# Licence:      The MIT License
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt


class ComlinkChannelBase(object):
    """Base class for holding CML channel data and metadata"""

    def __eq__(self):
        pass

    def __getitem__(self, key):
        new_cml_ch = self.__copy__()
        new_cml_ch.data = self.data.__getitem__(key)
        return new_cml_ch

    def __len__(self):
        return len(self.data)

    def __str__(self, *args, **kwargs):
        print 'f_GHz: ', self.f_GHz
        print self.data.__str__()

    def __getattr__(self, item):
        try:
            return self.data.__getattr__(item)
        except:
            raise AttributeError('Neither \'ComlinkChannel\' nor its '
                                 '\'DataFrame\' have the attribute \'%s\''
                                 % item)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __copy__(self):
        cls = self.__class__
        new_cml_ch = cls.__new__(cls)
        new_cml_ch.__dict__.update(self.__dict__)
        return new_cml_ch

    def __deepcopy__(self, memo=None):
        new_cml_ch = self.__copy__()
        if memo is None:
            memo = {}
        memo[id(self)] = new_cml_ch
        new_cml_ch.data = copy.deepcopy(self.data, memo)
        return new_cml_ch

    def _repr_html_(self):
        metadata_str = ''
        for key, value in self.metadata.iteritems():
            if key == 'frequency':
                metadata_str += (str(key) + ': ' + str(value/1e9) + ' GHz<br/>')
            else:
                metadata_str += (str(key) + ': ' + str(value) + '<br/>')
        return metadata_str + self.data._repr_html_()

    def copy(self):
        """ Return a deepcopy of this channel object """
        return self.__deepcopy__()

    def plot_data(self, columns=['rx', ], ax=None, channel_number=0):
        """ Plot time series of data from the different channels

        Linked subplots will be created for the different specified columns
        of the DataFrames of the different channels.

        Parameters
        ----------

        columns : list, optional
            List of DataFrame columns to plot for each channel.
            Defaults to ['rx', ]

        channels : list, optional
            List of channel names, i.e. the keys of the Comlink.channels
            dictionary, to specify which channel to plot. Defaults to None,
            which plots for all channels

        ax : matplotlib.axes, optional
            Axes handle, defaults to None, which plots into a new figure

        Returns
        -------

        ax : matplotlib.axes

        """
        if ax is None:
            fig, ax = plt.subplots(len(columns),
                                   1,
                                   figsize=(10, 1.5*len(columns) + 1),
                                   sharex=True)
        try:
            ax[0].get_alpha()
        except TypeError:
            ax = [ax, ]

        for ax_i, column in zip(ax, columns):
            if column == 'wet':
                ax_i.fill_between(
                    self.data[column].index,
                    channel_number,
                    channel_number + self.data[column].values,
                    alpha=0.9,
                    linewidth=0.0,
                    label=self.metadata['channel_id'])
            else:
                try:
                    self.data[column]
                    columns_to_plot = [column,]
                    alphas = [0.9,]
                except KeyError:
                    try:
                        self.data[column + '_min']
                        self.data[column + '_max']
                        columns_to_plot = [column + '_min', column + '_max']
                        alphas = [0.5, 0.9]
                    except Exception, e:
                        raise e

                line_color = None
                for column_i, alpha_i in zip(columns_to_plot, alphas):
                    if line_color is None:
                        line = ax_i.plot(
                            self.data[column_i].index,
                            self.data[column_i].values,
                            alpha=alpha_i,
                            label=self.metadata['channel_id'])
                        line_color = line[0].get_color()
                    else:
                        ax_i.plot(
                            self.data[column_i].index,
                            self.data[column_i].values,
                            color=line_color,
                            alpha=alpha_i,
                            label=self.metadata['channel_id'])
            ax_i.set_ylabel(column)

        return ax

    def resample(self, resampling_time, how=np.mean, inplace=False):
        """ Resample channel data

        Parameters
        ----------

        resampling_time : str
            The frequency to which you want to resample. Use the pandas
            notation, e.g. '5min' for 5 minutes or '3H' for three hours.

        how : function, optional
            The function to be applied for resampling. Defaults to `np.mean`,
            but e.g. also `np.max`, `np.min` or `np.sum` can make sense,
            depending on what you want

        inplace : bool, optional
            If set to `True` the resampling will be carried out directly on
            this `ComlinkChannel`. If set to `False`  a copy of the current
            `ComlinkChannel` with the resampled data will be returned. The
            original channel and its data will not be altered.

        Example
        -------

        # Resample an existing channel to 5 minutes
        cml_ch_5min = cml_ch_1min.resample('5min', inplace=False, how=np.mean)

        """

        if inplace:
            self.data = self.data.resample(resampling_time).apply(how)
        elif not inplace:
            new_cml_ch = copy.copy(self)
            new_cml_ch.data = self.data.resample(resampling_time).apply(how)
            return new_cml_ch
        else:
            raise ValueError('`inplace` must be either True or False')

    def _parse_metadata(self,
                        metadata_dict=None,
                        frequency=None, polarization=None,
                        atpc=None, channel_id=None):
        # TODO: Sanity check of metadata
        if metadata_dict is not None:
            self.metadata = metadata_dict
        else:
            self.metadata = {
                'frequency': frequency,
                'polarization': polarization,
                'channel_id': channel_id,
                'atpc': atpc}


class ComlinkChannel(ComlinkChannelBase):
    """A class for holding CML channel data and metadata"""

    def __init__(self, data=None, metadata=None,
                 t=None, rx=None, tx=None,
                 frequency=None, polarization=None,
                 atpc=None, channel_id=None
                 ):
        """

        Parameters
        ----------

        data: pandas.DataFrame
            DataFrame with the columns `tx` and `rx` holding the time series
            of the TX and RX level, respectively. The index of the DataFrame
            must contain the time stamps. If the TX level is constant,
            please still supply of full time series for it. You can specify
            that the TX level is constant by passing `atpc = 'off'`.

        t: list, np.array, or everything that DataFrame.__init__() digest

        rx: list or np.array, or everything that DataFrame.__init__() digest
            Timer series of RX power.

        tx: list, np.array, float or int
            Timer series of TX power. If only a scalar value is supplied,
            it is interpreted as the constant TX power.

        frequency: float
            Frequency in Hz.

        polarization: str {'h', 'v', 'H', 'V'}
            Polarization

        atpc: str {'on', 'off'}
            Flag to specifiy if ATPC (Automatic Transmission Power Control),
            i.e. a variable TX level, is active or not

        channel_id: str
            The ID of this channel.

        metadata: dict
           Dictionary with metadata, where this is an example of the minimum
           of information that has to be supplied in the dict, if it is not
           supplied seperately

           {'frequency': 20 * 1e9,
            'polarization': 'V',
            'channel_id': 'channel_xy'
            'atpc': 'off'}

        """

        # TODO: If this is not supplied we should maybe derive it somehow
        self.sampling_type = None

        # Handle the different arguments and build a DataFrame from them
        # if it has not been supplied as `data`
        self.data = _parse_kwargs_to_dataframe(data=data, t=t, rx=rx, tx=tx)
        # Check if the data is there as expected
        self.data.rx
        self.data.tx

        self._parse_metadata(metadata_dict=metadata,
                             frequency=frequency,
                             polarization=polarization,
                             channel_id=channel_id,
                             atpc=atpc)

        # TODO: Remove this
        # Keeping this for now for backwards compatibility
        self.f_GHz = self.metadata['frequency'] / 1e9


class ComlinkChannelMinMax(ComlinkChannelBase):
    """A class for holding CML channel data and metadata"""

    def __init__(self, data=None,
                 t=None,
                 rx_min=None, rx_max=None,
                 tx_min=None, tx_max=None, tx_const=None,
                 metadata=None,
                 frequency=None, polarization=None,
                 atpc=None, channel_id=None):
        """

        Parameters
        ----------

        data: pandas.DataFrame
            DataFrame with the columns `tx` and `rx` holding the time series
            of the TX and RX level, respectively. The index of the DataFrame
            must contain the time stamps. If the TX level is constant,
            please still supply of full time series for it. You can specify
            that the TX level is constant by passing `atpc = 'off'`.

        t: list, np.array, or everything that DataFrame.__init__() digest

        rx_min: list or np.array, or everything that DataFrame.__init__() digest
            Timer series of minimum RX power.

        rx_max: list or np.array, or everything that DataFrame.__init__() digest
            Timer series of maximum RX power.

        tx_min: list, np.array, float or int
            Timer series of minimum TX power.

        tx_max: list, np.array, float or int
            Timer series of minimum TX power.

        tx_const: float or int
            Constant TX power, which will be used for the whole time series

        frequency: float
            Frequency in Hz.

        polarization: str {'h', 'v', 'H', 'V'}
            Polarization

        atpc: str {'on', 'off'}
            Flag to specifiy if ATPC (Automatic Transmission Power Control),
            i.e. a variable TX level, is active or not

        channel_id: str
            The ID of this channel.

        metadata: dict
           Dictionary with metadata, where this is an example of the minimum
           of information that has to be supplied in the dict, if it is not
           supplied seperately

           {'frequency': 20 * 1e9,
            'polarization': 'V',
            'channel_id': 'channel_xy'
            'atpc': 'off'}

        """

        self.sampling_type = 'min_max'

        # Handle the different arguments and build a DataFrame from them
        # if it has not been supplied as `data`
        self.data = _parse_kwargs_to_dataframe(data=data,
                                               t=t,
                                               rx_min=rx_min, rx_max=rx_max,
                                               tx_min=tx_min, tx_max=tx_max,
                                               tx_const=tx_const)
        # Check if the data is there as expected
        self.data.rx_min
        self.data.rx_max
        self.data.tx_min
        self.data.tx_max

        self._parse_metadata(metadata_dict=metadata,
                             frequency=frequency,
                             polarization=polarization,
                             channel_id=channel_id,
                             atpc=atpc)

        # TODO: Remove this
        # Keeping this for now for backwards compatibility
        self.f_GHz = self.metadata['frequency'] / 1e9


def _parse_kwargs_to_dataframe(data, **kwargs):
    min_max_levels = None
    # The case where only `t`, `rx` and `tx` are supplied
    if data is None:
        # Check that the right kwargs have been supplied
        if set(['t', 'tx', 'rx']) == set(kwargs):
            min_max_levels = False
        elif (set(['t', 'tx_min', 'tx_max', 'rx_min', 'rx_max', 'tx_const']) ==
              set(kwargs)):
            min_max_levels = True
        else:
            raise ValueError('The kwargs for time, the RX- and TX-levels have '
                             'not been supplied correctly')

        if not min_max_levels:
            df = pd.DataFrame(index=kwargs['t'],
                              data={'rx': kwargs['rx']})
            df['tx'] = kwargs['tx']
        elif min_max_levels:
            df = pd.DataFrame(index=kwargs['t'],
                              data={'rx_min': kwargs['rx_min'],
                                    'rx_max': kwargs['rx_max']})
            if kwargs['tx_const'] is None:
                df['tx_min'] = kwargs['tx_min']
                df['tx_max'] = kwargs['tx_max']
            elif isinstance(kwargs['tx_const'], (int, float)):
                df['tx_min'] = kwargs['tx_const']
                df['tx_max'] = kwargs['tx_const']
            else:
                raise ValueError('`tx_const` must be None, int or float')

    # The case where `data` has been supplied.
    # We check that `data` is a DataFrame and that the DataFrame has the
    # columns `tx` and `rx`.
    elif isinstance(data, pd.DataFrame):
        # Check that the right column names have been supplied
        if set(['tx', 'rx']) == set(data.columns):
            min_max_levels = False
        elif (set(['tx_min', 'tx_max', 'rx_min', 'rx_max']) ==
              set(data.columns)):
            min_max_levels = True
        else:
            raise ValueError('`data` must have the correct names for the RX-'
                             'and TX-level columns')
        df = copy.deepcopy(data)

    else:
        raise ValueError('Could not parse the supplied arguments')

    if not min_max_levels:
        df['txrx'] = df.tx - df.rx

    df.index.name = 'time'

    return df
