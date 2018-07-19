# encoding: UTF-8

__version__ = '1.0'
__author__ = 'Andy Shao'

from .TraderAccount import TraderAccount
from .uiWidget import EngineManager

appName = 'vnApp'
appDisplayName = u'A股策略'
appEngine = TraderAccount
appWidget = EngineManager
appIco = 'ash.ico'