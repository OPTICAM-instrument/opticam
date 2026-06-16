from typing import Literal


from matplotlib.axes import Axes




def scale_ax(
    ax: Axes,
    scale: Literal['linear', 'semilogx', 'semilogy', 'loglog'],
    ) -> None:
    """
    Set the scale(s) of an `Axes` based on `scale`.

    Parameters
    ----------
    ax : Axes
        The axis to be scaled.
    scale : Literal[&#39;linear&#39;, &#39;semilogx&#39;, &#39;semilogy&#39;, &#39;loglog&#39;]
        The desired scale.
    """
    
    if scale == 'linear':
        return
    if scale == 'semilogx' or scale == 'loglog':
        ax.set_xscale('log')
    if scale == 'semilogy' or scale == 'loglog':
        ax.set_yscale('log')