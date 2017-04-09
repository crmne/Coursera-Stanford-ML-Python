# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import use

use('TkAgg')


def show():
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("+0+0")
    plt.show(block=False)
    wm.window.attributes('-topmost', 1)
