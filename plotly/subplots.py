from __future__ import absolute_import
import copy
import re

# Constants
# ---------

# Trace types that are individually positioned with their own domain.
# These are traces that don't overlay on top of each other in a shared subplot,
# so they are positioned individually.  All other trace types are associated
# with a layout subplot type (xaxis/yaxis, polar, scene etc.)
#
# Each of these trace types has a `domain` property with `x`/`y` properties
_domain_trace_types = {'parcoords', 'pie', 'table', 'sankey', 'parcats'}

# Subplot types that are each individually positioned with a domain
#
# Each of these subplot types has a `domain` property with `x`/`y` properties.
# Note that this set does not contain `xaxis`/`yaxis` because these behave a
# little differently.
_subplot_types = {'scene', 'geo', 'polar', 'ternary', 'mapbox'}

# For most subplot types, a trace is associated with a particular subplot
# using a trace property with a name that matches the subplot type. For
# example, a `scatter3d.scene` property set to `'scene2'` associates a
# scatter3d trace with the second `scene` subplot in the figure.
#
# There are a few subplot types that don't follow this pattern, and instead
# the trace property is just named `subplot`.  For example setting
# the `scatterpolar.subplot` property to `polar3` associates the scatterpolar
# trace with the third polar subplot in the figure
_subplot_prop_named_subplot = {'polar', 'ternary', 'mapbox'}

# Mapping from trace type to subplot type(s).
_trace_to_subplot = {
    # xaxis/yaxis
    'bar':                  ['xaxis', 'yaxis'],
    'box':                  ['xaxis', 'yaxis'],
    'candlestick':          ['xaxis', 'yaxis'],
    'carpet':               ['xaxis', 'yaxis'],
    'contour':              ['xaxis', 'yaxis'],
    'contourcarpet':        ['xaxis', 'yaxis'],
    'heatmap':              ['xaxis', 'yaxis'],
    'heatmapgl':            ['xaxis', 'yaxis'],
    'histogram':            ['xaxis', 'yaxis'],
    'histogram2d':          ['xaxis', 'yaxis'],
    'histogram2dcontour':   ['xaxis', 'yaxis'],
    'ohlc':                 ['xaxis', 'yaxis'],
    'pointcloud':           ['xaxis', 'yaxis'],
    'scatter':              ['xaxis', 'yaxis'],
    'scattercarpet':        ['xaxis', 'yaxis'],
    'scattergl':            ['xaxis', 'yaxis'],
    'violin':               ['xaxis', 'yaxis'],

    # scene
    'cone':         ['scene'],
    'mesh3d':       ['scene'],
    'scatter3d':    ['scene'],
    'streamtube':   ['scene'],
    'surface':      ['scene'],

    # geo
    'choropleth': ['geo'],
    'scattergeo': ['geo'],

    # polar
    'barpolar':         ['polar'],
    'scatterpolar':     ['polar'],
    'scatterpolargl':   ['polar'],

    # ternary
    'scatterternary': ['ternary'],

    # mapbox
    'scattermapbox': ['mapbox']
}

# Regular expression to extract any trailing digits from a subplot-style
# string.
_subplot_re = re.compile('\D*(\d+)')


def _get_subplot_number(subplot_val):
    """
    Extract the subplot number from a subplot value string.

    'x3' -> 3
    'polar2' -> 2
    'scene' -> 1
    'y' -> 1

    Note: the absence of a subplot number (e.g. 'y') is treated by plotly as
    a subplot number of 1

    Parameters
    ----------
    subplot_val: str
        Subplot string value (e.g. 'scene4')

    Returns
    -------
    int
    """
    match = _subplot_re.match(subplot_val)
    if match:
        subplot_number = int(match.group(1))
    else:
        subplot_number = 1
    return subplot_number


def _get_subplot_val_prefix(subplot_type):
    """
    Get the subplot value prefix for a subplot type. For most subplot types
    this is equal to the subplot type string itself. For example, a
    `scatter3d.scene` value of `scene2` is used to associate the scatter3d
    trace with the `layout.scene2` subplot.

    However, the `xaxis`/`yaxis` subplot types are exceptions to this pattern.
    For example, a `scatter.xaxis` value of `x2` is used to associate the
    scatter trace with the `layout.xaxis2` subplot.

    Parameters
    ----------
    subplot_type: str
        Subplot string value (e.g. 'scene4')

    Returns
    -------
    str
    """
    if subplot_type == 'xaxis':
        subplot_val_prefix = 'x'
    elif subplot_type == 'yaxis':
        subplot_val_prefix = 'y'
    else:
        subplot_val_prefix = subplot_type
    return subplot_val_prefix


def _get_subplot_prop_name(subplot_type):
    """
    Get the name of the trace property used to associate a trace with a
    particular subplot type.  For most subplot types this is equal to the
    subplot type string. For example, the `scatter3d.scene` property is used
    to associate a `scatter3d` trace with a particular `scene` subplot.

    However, for some subplot types the trace property is not named after the
    subplot type.  For example, the `scatterpolar.subplot` property is used
    to associate a `scatterpolar` trace with a particular `polar` subplot.


    Parameters
    ----------
    subplot_type: str
        Subplot string value (e.g. 'scene4')

    Returns
    -------
    str
    """
    if subplot_type in _subplot_prop_named_subplot:
        subplot_prop_name = 'subplot'
    else:
        subplot_prop_name = subplot_type
    return subplot_prop_name


def _normalize_subplot_ids(fig):
    """
    Make sure a layout subplot property is initialized for every subplot that
    is referenced by a trace in the figure.

    For example, if a figure contains a `scatterpolar` trace with the `subplot`
    property set to `polar3`, this function will make sure the figure's layout
    has a `polar3` property, and will initialize it to an empty dict if it
    does not

    Note: This function mutates the input figure dict

    Parameters
    ----------
    fig: dict
        A plotly figure dict
    """

    layout = fig.setdefault('layout', {})
    for trace in fig.get('data', None):
        trace_type = trace.get('type', 'scatter')
        subplot_types = _trace_to_subplot.get(trace_type, [])
        for subplot_type in subplot_types:

            subplot_prop_name = _get_subplot_prop_name(subplot_type)
            subplot_val_prefix = _get_subplot_val_prefix(subplot_type)
            subplot_val = trace.get(subplot_prop_name, subplot_val_prefix)

            # extract trailing number (if any)
            subplot_number = _get_subplot_number(subplot_val)

            if subplot_number > 1:
                layout_prop_name = subplot_type + str(subplot_number)
            else:
                layout_prop_name = subplot_type

            if layout_prop_name not in layout:
                layout[layout_prop_name] = {}


def _get_max_subplot_ids(fig):
    """
    Given an input figure, return a dict containing the max subplot number
    for each subplot type in the figure

    Parameters
    ----------
    fig: dict
        A plotly figure dict

    Returns
    -------
    dict
        A dict from subplot type strings to integers indicating the largest
        subplot number in the figure of that subplot type
    """
    max_subplot_ids = _get_initial_max_subplot_ids()

    for trace in fig.get('data', []):
        trace_type = trace.get('type', 'scatter')
        subplot_types = _trace_to_subplot.get(trace_type, [])
        for subplot_type in subplot_types:

            subplot_prop_name = _get_subplot_prop_name(subplot_type)
            subplot_val_prefix = _get_subplot_val_prefix(subplot_type)
            subplot_val = trace.get(subplot_prop_name, subplot_val_prefix)

            # extract trailing number (if any)
            subplot_number = _get_subplot_number(subplot_val)

            max_subplot_ids[subplot_type] = max(
                max_subplot_ids[subplot_type], subplot_number)

    return max_subplot_ids


def _get_initial_max_subplot_ids():
    max_subplot_ids = {subplot_type: 0
                       for subplot_type in _subplot_types}
    max_subplot_ids['xaxis'] = 0
    max_subplot_ids['yaxis'] = 0
    return max_subplot_ids


def _offset_subplot_ids(fig, offsets):
    """
    Apply offsets to the subplot id numbers in a figure.

    Note: This function mutates the input figure dict

    Note: This function assumes that the normalize_subplot_ids function has
    already been run on the figure, so that all layout subplot properties in
    use are explicitly present in the figure's layout.

    Parameters
    ----------
    fig: dict
        A plotly figure dict
    offsets: dict
        A dict from subplot types to the offset to be applied for each subplot
        type.  This dict matches the form of the dict returned by
        get_max_subplot_ids
    """
    # Offset traces
    for trace in fig.get('data', None):
        trace_type = trace.get('type', 'scatter')
        subplot_types = _trace_to_subplot.get(trace_type, [])

        for subplot_type in subplot_types:
            subplot_prop_name = _get_subplot_prop_name(subplot_type)

            # Compute subplot value prefix
            subplot_val_prefix = _get_subplot_val_prefix(subplot_type)
            subplot_val = trace.get(subplot_prop_name, subplot_val_prefix)
            subplot_number = _get_subplot_number(subplot_val)

            offset_subplot_number = (
                    subplot_number + offsets.get(subplot_type, 0))

            if offset_subplot_number > 1:
                trace[subplot_prop_name] = (
                        subplot_val_prefix + str(offset_subplot_number))
            else:
                trace[subplot_prop_name] = subplot_val_prefix

    # layout subplots
    layout = fig.setdefault('layout', {})
    new_subplots = {}

    for subplot_type in offsets:
        offset = offsets[subplot_type]
        if offset < 1:
            continue

        for layout_prop in list(layout.keys()):
            if layout_prop.startswith(subplot_type):
                subplot_number = _get_subplot_number(layout_prop)
                new_subplot_number = subplot_number + offset
                new_layout_prop = subplot_type + str(new_subplot_number)
                new_subplots[new_layout_prop] = layout.pop(layout_prop)

    layout.update(new_subplots)

    # xaxis/yaxis anchors
    x_offset = offsets.get('xaxis', 0)
    y_offset = offsets.get('yaxis', 0)

    for layout_prop in list(layout.keys()):
        if layout_prop.startswith('xaxis'):
            xaxis = layout[layout_prop]
            anchor = xaxis.get('anchor', 'y')
            anchor_number = _get_subplot_number(anchor) + y_offset
            if anchor_number > 1:
                xaxis['anchor'] = 'y' + str(anchor_number)
            else:
                xaxis['anchor'] = 'y'
        elif layout_prop.startswith('yaxis'):
            yaxis = layout[layout_prop]
            anchor = yaxis.get('anchor', 'x')
            anchor_number = _get_subplot_number(anchor) + x_offset
            if anchor_number > 1:
                yaxis['anchor'] = 'x' + str(anchor_number)
            else:
                yaxis['anchor'] = 'x'

    # annotations/shapes/images
    for layout_prop in ['annotations', 'shapes', 'images']:
        for obj in layout.get(layout_prop, []):
            if x_offset:
                xref = obj.get('xref', 'x')
                if xref != 'paper':
                    xref_number = _get_subplot_number(xref)
                    obj['xref'] = 'x' + str(xref_number + x_offset)

            if y_offset:
                yref = obj.get('yref', 'y')
                if yref != 'paper':
                    yref_number = _get_subplot_number(yref)
                    obj['yref'] = 'y' + str(yref_number + y_offset)


def _scale_translate(fig, scale_x, scale_y, translate_x, translate_y):
    """
    Scale a figure and translate it to sub-region of the original
    figure canvas.

    Note: If the input figure has a title, this title is converted into an
    annotation and scaled along with the rest of the figure.

    Note: This function mutates the input fig dict

    Note: This function assumes that the normalize_subplot_ids function has
    already been run on the figure, so that all layout subplot properties in
    use are explicitly present in the figure's layout.

    Parameters
    ----------
    fig: dict
        A plotly figure dict
    scale_x: float
        Factor by which to scale the figure in the x-direction. This will
        typically be a value < 1.  E.g. a value of 0.5 will cause the
        resulting figure to be half as wide as the original.
    scale_y: float
        Factor by which to scale the figure in the y-direction. This will
        typically be a value < 1
    translate_x: float
        Factor by which to translate the scaled figure in the x-direction in
        normalized coordinates.
    translate_y: float
        Factor by which to translate the scaled figure in the x-direction in
        normalized coordinates.
    """
    data = fig.setdefault('data', [])
    layout = fig.setdefault('layout', {})

    def scale_translate_x(x):
        return [x[0] * scale_x + translate_x,
                x[1] * scale_x + translate_x]

    def scale_translate_y(y):
        return [y[0] * scale_y + translate_y,
                y[1] * scale_y + translate_y]

    def perform_scale_translate(obj):
        domain = obj.setdefault('domain', {})
        x = domain.get('x', [0, 1])
        y = domain.get('y', [0, 1])

        domain['x'] = scale_translate_x(x)
        domain['y'] = scale_translate_y(y)

    # Scale/translate traces
    for trace in data:
        trace_type = trace.get('type', 'scatter')
        if trace_type in _domain_trace_types:
            perform_scale_translate(trace)

    # Scale/translate subplot containers
    for prop in layout:
        for subplot_type in _subplot_types:
            if prop.startswith(subplot_type):
                perform_scale_translate(layout[prop])

    for prop in layout:
        if prop.startswith('xaxis'):
            xaxis = layout[prop]
            x_domain = xaxis.get('domain', [0, 1])
            xaxis['domain'] = scale_translate_x(x_domain)
        elif prop.startswith('yaxis'):
            yaxis = layout[prop]
            y_domain = yaxis.get('domain', [0, 1])
            yaxis['domain'] = scale_translate_y(y_domain)

    # convert title to annotation
    # This way the annotation will be scaled with the reset of the figure
    annotations = layout.get('annotations', [])

    title = layout.pop('title', None)
    if title:
        titlefont = layout.pop('titlefont', {})
        title_fontsize = titlefont.get('size', 17)
        min_fontsize = 12
        titlefont['size'] = round(min_fontsize +
                                  (title_fontsize - min_fontsize) * scale_x)

        annotations.append({
            'text': title,
            'showarrow': False,
            'xref': 'paper',
            'yref': 'paper',
            'x': 0.5,
            'y': 1.01,
            'xanchor': 'center',
            'yanchor': 'bottom',
            'font': titlefont
        })
        layout['annotations'] = annotations

    # annotations
    for obj in layout.get('annotations', []):
        if obj.get('xref', None) == 'paper':
            obj['x'] = obj.get('x', 0.5) * scale_x + translate_x
            obj['y'] = obj.get('y', 0.5) * scale_y + translate_y


def merge_figure(fig, subfig):
    """
    Merge a sub-figure into a parent figure

    Note: This function mutates the input fig dict, but it does not mutate
    the subfig dict

    Parameters
    ----------
    fig: dict
        The plotly figure dict into which the sub figure will be merged
    subfig: dict
        The plotly figure dict that will be copied and then merged into `fig`
    """

    # traces
    data = fig.setdefault('data', [])
    data.extend(copy.deepcopy(subfig.get('data', [])))

    # layout
    layout = fig.setdefault('layout', {})
    _merge_layout_objs(layout, subfig.get('layout', {}))


def _merge_layout_objs(obj, subobj):
    """
    Merge layout objects recursively

    Note: This function mutates the input obj dict, but it does not mutate
    the subobj dict

    Parameters
    ----------
    obj: dict
        dict into which the sub-figure dict will be merged
    subobj: dict
        dict that sill be copied and merged into `obj`
    """
    for prop, val in subobj.items():
        if isinstance(val, dict) and prop in obj:
            # recursion
            _merge_layout_objs(obj[prop], val)
        elif (isinstance(val, list) and
              obj.get(prop, None) and
              isinstance(obj[prop][0], dict)):

            # append
            obj[prop].extend(val)
        else:
            # init/overwrite
            obj[prop] = copy.deepcopy(val)


def _compute_subplot_domains(widths, spacing):
    """
    Compute normalized domain tuples for a list of widths and a subplot
    spacing value

    Parameters
    ----------
    widths: list of float
        List of the desired withs of each subplot. The length of this list
        is also the specification of the number of desired subplots
    spacing: float
        Spacing between subplots in normalized coordinates

    Returns
    -------
    list of tuple of float
    """
    # normalize widths
    widths_sum = float(sum(widths))
    total_spacing = (len(widths) - 1) * spacing
    widths = [(w / widths_sum)*(1-total_spacing) for w in widths]
    domains = []

    for c in range(len(widths)):
        domain_start = c * spacing + sum(widths[:c])
        domain_stop = min(1, domain_start + widths[c])
        domains.append((domain_start, domain_stop))

    return domains


def figure_grid(figures_grid,
                row_heights=None,
                column_widths=None,
                row_spacing=0.15,
                column_spacing=0.15,
                share_xaxis=False,
                share_yaxis=False):
    """
    Construct a figure from a 2D grid of sub-figures

    Parameters
    ----------
    figures_grid: list of list of (dict or None)
        2D list of plotly figure dicts that will be combined in a grid to
        produce the resulting figure.  None values maybe used to leave empty
        grid cells
    row_heights: list of float (default None)
        List of the relative heights of each row in the grid (these values
        will be normalized by the function)
    column_widths: list of float (default None)
        List of the relative widths of each column in the grid (these values
        will be normalized by the function)
    row_spacing: float (default 0.15)
        Vertical spacing between rows in the gird in normalized coordinates
    column_spacing: float (default 0.15)
        Horizontal spacing between columns in the grid in normalized
        coordinates
    share_xaxis: bool (default False)
        Share x-axis between sub-figures in the same column. This will only
        work if each sub-figure has a single x-axis
    share_yaxis: bool (default False)
        Share y-axis between sub-figures in the same row. This will only work
        if each subfigure has a single y-axis

    Returns
    -------
    dict
        A plotly figure dict
    """
    from plotly.basedatatypes import BaseFigure

    # compute number of rows/cols
    rows = len(figures_grid)
    columns = len(figures_grid[0])

    # Initialize row heights / column widths
    if not row_heights:
        row_heights = [1 for _ in range(rows)]

    if not column_widths:
        column_widths = [1 for _ in range(columns)]

    # Compute domain widths/heights for subplots
    column_domains = _compute_subplot_domains(column_widths, column_spacing)
    row_domains = _compute_subplot_domains(row_heights, row_spacing)

    output_figure = {'data': [], 'layout': {}}

    for r, (fig_row, row_domain) in enumerate(zip(figures_grid, row_domains)):
        for c, (fig, column_domain) in enumerate(zip(fig_row, column_domains)):
            if fig:
                if isinstance(fig, BaseFigure):
                    fig = fig.to_dict()
                else:
                    fig = copy.deepcopy(fig)

                _normalize_subplot_ids(fig)

                subplot_offsets = _get_max_subplot_ids(output_figure)

                if share_xaxis:
                    subplot_offsets['xaxis'] = c
                    if r != 0:
                        # Only use xaxes from bottom row
                        fig.get('layout', {}).pop('xaxis', None)

                if share_yaxis:
                    subplot_offsets['yaxis'] = r
                    if c != 0:
                        # Only use yaxes from first column
                        fig.get('layout', {}).pop('yaxis', None)

                _offset_subplot_ids(fig, subplot_offsets)

                scale_x = column_domain[1] - column_domain[0]
                scale_y = row_domain[1] - row_domain[0]
                _scale_translate(fig,
                                 scale_x, scale_y,
                                 column_domain[0], row_domain[0])

                merge_figure(output_figure, fig)

    return output_figure


def make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=False,
        shared_yaxes=False,
        start_cell='top-left',
        print_grid=True,
        horizontal_spacing=None,
        vertical_spacing=None,
        subplot_titles=None,
        column_width=None,
        row_width=None,
        specs=None,
        insets=None,
        column_titles=None,
        row_titles=None
):
    """Return an instance of plotly.graph_objs.Figure
    with the subplots domain set in 'layout'.

    Example 1:
    # stack two subplots vertically
    fig = tools.make_subplots(rows=2)

    This is the format of your plot grid:
    [ (1,1) x1,y1 ]
    [ (2,1) x2,y2 ]

    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]

    # or see Figure.append_trace

    Example 2:
    # subplots with shared x axes
    fig = tools.make_subplots(rows=2, shared_xaxes=True)

    This is the format of your plot grid:
    [ (1,1) x1,y1 ]
    [ (2,1) x1,y2 ]


    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], yaxis='y2')]

    Example 3:
    # irregular subplot layout (more examples below under 'specs')
    fig = tools.make_subplots(rows=2, cols=2,
                              specs=[[{}, {}],
                                     [{'colspan': 2}, None]])

    This is the format of your plot grid!
    [ (1,1) x1,y1 ]  [ (1,2) x2,y2 ]
    [ (2,1) x3,y3           -      ]

    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]
    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x3', yaxis='y3')]

    Example 4:
    # insets
    fig = tools.make_subplots(insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.3}])

    This is the format of your plot grid!
    [ (1,1) x1,y1 ]

    With insets:
    [ x2,y2 ] over [ (1,1) x1,y1 ]

    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]

    Example 5:
    # include subplot titles
    fig = tools.make_subplots(rows=2, subplot_titles=('Plot 1','Plot 2'))

    This is the format of your plot grid:
    [ (1,1) x1,y1 ]
    [ (2,1) x2,y2 ]

    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]

    Example 6:
    # Include subplot title on one plot (but not all)
    fig = tools.make_subplots(insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.3}],
                              subplot_titles=('','Inset'))

    This is the format of your plot grid!
    [ (1,1) x1,y1 ]

    With insets:
    [ x2,y2 ] over [ (1,1) x1,y1 ]

    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]

    Keywords arguments with constant defaults:

    rows (kwarg, int greater than 0, default=1):
        Number of rows in the subplot grid.

    cols (kwarg, int greater than 0, default=1):
        Number of columns in the subplot grid.

    shared_xaxes (kwarg, boolean or list, default=False)
        Assign shared x axes.
        If True, subplots in the same grid column have one common
        shared x-axis at the bottom of the gird.

        To assign shared x axes per subplot grid cell (see 'specs'),
        send list (or list of lists, one list per shared x axis)
        of cell index tuples.

    shared_yaxes (kwarg, boolean or list, default=False)
        Assign shared y axes.
        If True, subplots in the same grid row have one common
        shared y-axis on the left-hand side of the gird.

        To assign shared y axes per subplot grid cell (see 'specs'),
        send list (or list of lists, one list per shared y axis)
        of cell index tuples.

    start_cell (kwarg, 'bottom-left' or 'top-left', default='top-left')
        Choose the starting cell in the subplot grid used to set the
        domains_grid of the subplots.

    print_grid (kwarg, boolean, default=True):
        If True, prints a tab-delimited string representation of
        your plot grid.

    Keyword arguments with variable defaults:

    horizontal_spacing (kwarg, float in [0,1], default=0.2 / cols):
        Space between subplot columns.
        Applies to all columns (use 'specs' subplot-dependents spacing)

    vertical_spacing (kwarg, float in [0,1], default=0.3 / rows):
        Space between subplot rows.
        Applies to all rows (use 'specs' subplot-dependents spacing)

    subplot_titles (kwarg, list of strings, default=empty list):
        Title of each subplot.
        "" can be included in the list if no subplot title is desired in
        that space so that the titles are properly indexed.

    specs (kwarg, list of lists of dictionaries):
        Subplot specifications.

        ex1: specs=[[{}, {}], [{'colspan': 2}, None]]

        ex2: specs=[[{'rowspan': 2}, {}], [None, {}]]

        - Indices of the outer list correspond to subplot grid rows
          starting from the bottom. The number of rows in 'specs'
          must be equal to 'rows'.

        - Indices of the inner lists correspond to subplot grid columns
          starting from the left. The number of columns in 'specs'
          must be equal to 'cols'.

        - Each item in the 'specs' list corresponds to one subplot
          in a subplot grid. (N.B. The subplot grid has exactly 'rows'
          times 'cols' cells.)

        - Use None for blank a subplot cell (or to move pass a col/row span).

        - Note that specs[0][0] has the specs of the 'start_cell' subplot.

        - Each item in 'specs' is a dictionary.
            The available keys are:

            * is_3d (boolean, default=False): flag for 3d scenes
            * colspan (int, default=1): number of subplot columns
                for this subplot to span.
            * rowspan (int, default=1): number of subplot rows
                for this subplot to span.
            * l (float, default=0.0): padding left of cell
            * r (float, default=0.0): padding right of cell
            * t (float, default=0.0): padding right of cell
            * b (float, default=0.0): padding bottom of cell

        - Use 'horizontal_spacing' and 'vertical_spacing' to adjust
          the spacing in between the subplots.

    insets (kwarg, list of dictionaries):
        Inset specifications.

        - Each item in 'insets' is a dictionary.
            The available keys are:

            * cell (tuple, default=(1,1)): (row, col) index of the
                subplot cell to overlay inset axes onto.
            * is_3d (boolean, default=False): flag for 3d scenes
            * l (float, default=0.0): padding left of inset
                  in fraction of cell width
            * w (float or 'to_end', default='to_end') inset width
                  in fraction of cell width ('to_end': to cell right edge)
            * b (float, default=0.0): padding bottom of inset
                  in fraction of cell height
            * h (float or 'to_end', default='to_end') inset height
                  in fraction of cell height ('to_end': to cell top edge)

    column_width (kwarg, list of numbers)
        Column_width specifications

        - Functions similarly to `column_width` of `plotly.graph_objs.Table`.
          Specify a list that contains numbers where the amount of numbers in
          the list is equal to `cols`.

        - The numbers in the list indicate the proportions that each column
          domains_grid take across the full horizontal domain excluding padding.

        - For example, if columns_width=[3, 1], horizontal_spacing=0, and
          cols=2, the domains_grid for each column would be [0. 0.75] and [0.75, 1]

    row_width (kwargs, list of numbers)
        Row_width specifications

        - Functions similarly to `column_width`. Specify a list that contains
          numbers where the amount of numbers in the list is equal to `rows`.

        - The numbers in the list indicate the proportions that each row
          domains_grid take along the full vertical domain excluding padding.

        - For example, if row_width=[3, 1], vertical_spacing=0, and
          cols=2, the domains_grid for each row from top to botton would be
          [0. 0.75] and [0.75, 1]
    """
    import plotly.graph_objs as go

    # Validate coerce inputs
    # ----------------------
    #  ### rows ###
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError("""
The 'rows' argument to make_suplots must be an int greater than 0.
    Received value of type {typ}: {val}""".format(
            typ=type(rows), val=repr(rows)))

    #  ### cols ###
    if not isinstance(cols, int) or cols <= 0:
        raise ValueError("""
The 'cols' argument to make_suplots must be an int greater than 0.
    Received value of type {typ}: {val}""".format(
            typ=type(cols), val=repr(cols)))

    # ### start_cell ###
    if start_cell == 'bottom-left':
        col_dir = 1
        row_dir = 1
    elif start_cell == 'top-left':
        col_dir = 1
        row_dir = -1
    else:
        raise ValueError("""
The 'start_cell` argument to make_subplots must be one of \
['bottom-left', 'top-left']
    Received value of type {typ}: {val}""".format(
            typ=type(start_cell), val=repr(start_cell)))

    # ### horizontal_spacing ###
    if horizontal_spacing is None:
        horizontal_spacing = 0.2 / cols

    # ### vertical_spacing ###
    if vertical_spacing is None:
        if subplot_titles:
            vertical_spacing = 0.5 / rows
        else:
            vertical_spacing = 0.3 / rows

    # ### subplot titles ###
    if not subplot_titles:
        subplot_titles = [""] * rows * cols

    # ### column_width ###
    if row_titles:
        # Add a little breathing room between row labels and legend
        max_width = 0.98
    else:
        max_width = 1.0
    if column_width is None:
        widths = [(max_width - horizontal_spacing * (cols - 1)) / cols] * cols
    elif isinstance(column_width, (list, tuple)) and len(column_width) == cols:
        cum_sum = float(sum(column_width))
        widths = []
        for w in column_width:
            widths.append(
                (max_width - horizontal_spacing * (cols - 1)) * (w / cum_sum)
            )
    else:
        raise ValueError("""
The 'column_width' argument to make_suplots must be a list of numbers of \
length {cols}.
    Received value of type {typ}: {val}""".format(
            cols=cols, typ=type(cols), val=repr(column_width)))

    # ### row_width ###
    if row_width is None:
        heights = [(1. - vertical_spacing * (rows - 1)) / rows] * rows
    elif isinstance(row_width, (list, tuple)) and len(row_width) == rows:
        cum_sum = float(sum(row_width))
        heights = []
        for h in row_width:
            heights.append(
                (1. - vertical_spacing * (rows - 1)) * (h / cum_sum)
            )
        if row_dir < 0:
            heights = list(reversed(heights))
    else:
        raise ValueError("""
The 'row_width' argument to make_suplots must be a list of numbers of \
length {rows}.
    Received value of type {typ}: {val}""".format(
            rows=rows, typ=type(cols), val=repr(row_width)))

    # ### Helper to validate coerce elements of lists of dictionaries ###
    def _check_keys_and_fill(name, arg, defaults):
        def _checks(item, defaults):
            if item is None:
                return
            if not isinstance(item, dict):
                raise ValueError("""
Elements of the '{name}' argument to make_suplots must be dictionaries \
or None.
    Received value of type {typ}: {val}""".format(
                    name=name, typ=type(item), val=repr(item)
                ))

            for k in item:
                if k not in defaults:
                    raise ValueError("""
Invalid key specified in an element of the '{name}' argument to \
make_subplots: {k}
    Valid keys include: {valid_keys}""".format(
                        k=repr(k), name=name,
                        valid_keys=repr(list(defaults))
                    ))
            for k, v in defaults.items():
                item.setdefault(k, v)

        for arg_i in arg:
            if isinstance(arg_i, (list, tuple)):
                for arg_ii in arg_i:
                    _checks(arg_ii, defaults)
            elif isinstance(arg_i, dict):
                _checks(arg_i, defaults)

    # ### specs ###
    if specs is None:
        specs = [[{} for c in range(cols)] for r in range(rows)]
    elif not (
            isinstance(specs, (list, tuple))
            and specs
            and all(isinstance(row, (list, tuple)) for row in specs)
            and len(specs) == rows
            and all(len(row) == cols for row in specs)
            and all(all(v is None or isinstance(v, dict)
                        for v in row)
                    for row in specs)
    ):
        raise ValueError("""
The 'specs' argument to make_subplots must be a 2D list of dictionaries with \
dimensions ({rows} x {cols}).
    Received value of type {typ}: {val}""".format(
            rows=rows, cols=cols, typ=type(specs), val=repr(specs)
        ))

    # For backward compatibility, convert is_3d flag to type='3d' kwarg
    for row in specs:
        for spec in row:
            if spec and spec.pop('is_3d', None):
                spec['type'] = '3d'

    spec_defaults = dict(
        type='2d',
        colspan=1,
        rowspan=1,
        l=0.0,
        r=0.0,
        b=0.0,
        t=0.0
    )
    _check_keys_and_fill('specs', specs, spec_defaults)

    # ### insets ###
    if insets is None or insets is False:
        insets = False
    elif not (
            isinstance(insets, (list, tuple)) and
            all(isinstance(v, dict) for v in insets)
    ):
        raise ValueError("""
The 'insets' argument to make_suplots must be a list of dictionaries.
    Received value of type {typ}: {val}""".format(
            typ=type(insets), val=repr(insets)))

    if insets:
        # For backward compatibility, convert is_3d flag to type='3d' kwarg
        for inset in insets:
            if inset and inset.pop('is_3d', None):
                inset['type'] = '3d'

        inset_defaults = dict(
            cell=(1, 1),
            type='2d',
            l=0.0,
            w='to_end',
            b=0.0,
            h='to_end'
        )
        _check_keys_and_fill('insets', insets, inset_defaults)

    # Init layout
    # -----------
    layout = go.Layout()

    # Build grid reference
    # --------------------
    # Built row/col sequence using 'row_dir' and 'col_dir'
    col_seq = range(cols)[::col_dir]
    row_seq = range(rows)[::row_dir]

    # Build 2D array of tuples of the start x and start y coordinate of each
    # subplot
    grid = [
        [
            (
                (sum(widths[:c]) + c * horizontal_spacing),
                (sum(heights[:r]) + r * vertical_spacing)
            ) for c in col_seq
        ] for r in row_seq
    ]

    domains_grid = [[None for _ in range(cols)] for _ in range(rows)]

    # Initialize subplot reference lists for the grid and insets
    grid_ref = [[None for c in range(cols)] for r in range(rows)]

    list_of_domains = []  # added for subplot titles

    max_subplot_ids = _get_initial_max_subplot_ids()

    # Loop through specs -- (r, c) <-> (row, col)
    for r, spec_row in enumerate(specs):
        for c, spec in enumerate(spec_row):

            if spec is None:  # skip over None cells
                continue

            # ### Compute x and y domain for subplot ###
            c_spanned = c + spec['colspan'] - 1  # get spanned c
            r_spanned = r + spec['rowspan'] - 1  # get spanned r

            # Throw exception if 'colspan' | 'rowspan' is too large for grid
            if c_spanned >= cols:
                raise Exception("Some 'colspan' value is too large for "
                                "this subplot grid.")
            if r_spanned >= rows:
                raise Exception("Some 'rowspan' value is too large for "
                                "this subplot grid.")

            # Get x domain using grid and colspan
            x_s = grid[r][c][0] + spec['l']

            x_e = grid[r][c_spanned][0] + widths[c_spanned] - spec['r']
            x_domain = [x_s, x_e]

            # Get y domain (dep. on row_dir) using grid & r_spanned
            if row_dir > 0:
                y_s = grid[r][c][1] + spec['b']
                y_e = grid[r_spanned][c][1] + heights[r_spanned] - spec['t']
            else:
                y_s = grid[r_spanned][c][1] + spec['b']
                y_e = grid[r][c][1] + heights[-1 - r] - spec['t']
            y_domain = [y_s, y_e]

            list_of_domains.append(x_domain)
            list_of_domains.append(y_domain)

            domains_grid[r][c] = [x_domain, y_domain]

            # ### construct subplot container ###
            subplot_type = spec['type']
            grid_ref_element = _init_subplot(
                layout, subplot_type, x_domain, y_domain, max_subplot_ids)
            grid_ref_element['spec'] = spec
            grid_ref[r][c] = grid_ref_element

    _configure_shared_axes(layout, grid_ref, 'x', shared_xaxes, row_dir)
    _configure_shared_axes(layout, grid_ref, 'y', shared_yaxes, row_dir)

    # Build inset reference
    # ---------------------
    # Loop through insets
    insets_ref = [None for inset in range(len(insets))] if insets else None
    if insets:
        for i_inset, inset in enumerate(insets):

            r = inset['cell'][0] - 1
            c = inset['cell'][1] - 1

            # Throw exception if r | c is out of range
            if not (0 <= r < rows):
                raise Exception("Some 'cell' row value is out of range. "
                                "Note: the starting cell is (1, 1)")
            if not (0 <= c < cols):
                raise Exception("Some 'cell' col value is out of range. "
                                "Note: the starting cell is (1, 1)")

            # Get inset x domain using grid
            x_s = grid[r][c][0] + inset['l'] * widths[c]
            if inset['w'] == 'to_end':
                x_e = grid[r][c][0] + widths[c]
            else:
                x_e = x_s + inset['w'] * widths[c]
            x_domain = [x_s, x_e]

            # Get inset y domain using grid
            y_s = grid[r][c][1] + inset['b'] * heights[-1 - r]
            if inset['h'] == 'to_end':
                y_e = grid[r][c][1] + heights[-1 - r]
            else:
                y_e = y_s + inset['h'] * heights[-1 - r]
            y_domain = [y_s, y_e]

            subplot_type = inset['type']

            inset_ref_element = _init_subplot(
                layout, subplot_type, x_domain, y_domain, max_subplot_ids)

            insets_ref[i_inset] = inset_ref_element

    # Build grid_str
    # This is the message printed when print_grid=True
    grid_str = _build_grid_str(specs, grid_ref, insets, insets_ref, row_seq)

    # Add subplot titles
    plot_title_annotations = _build_subplot_title_annotations(
        subplot_titles,
        list_of_domains,
    )

    layout['annotations'] = plot_title_annotations

    # Add column titles
    if column_titles:
        domains_list = []
        if row_dir > 0:
            for c in range(cols):
                domain_pair = domains_grid[-1][c]
                if domain_pair:
                    domains_list.extend(domain_pair)
        else:
            for c in range(cols):
                domain_pair = domains_grid[0][c]
                if domain_pair:
                    domains_list.extend(domain_pair)

        # Add subplot titles
        column_title_annotations = _build_subplot_title_annotations(
            column_titles,
            domains_list,
        )

        layout['annotations'] += tuple(column_title_annotations)

    if row_titles:
        domains_list = []
        if row_dir < 0:
            rows_iter = range(rows - 1, -1, -1)
        else:
            rows_iter = range(rows)

        for r in rows_iter:
            domain_pair = domains_grid[r][-1]
            if domain_pair:
                domains_list.extend(domain_pair)

        # Add subplot titles
        column_title_annotations = _build_subplot_title_annotations(
            row_titles,
            domains_list,
            title_edge='right'
        )

        layout['annotations'] += tuple(column_title_annotations)


    # Handle displaying grid information
    if print_grid:
        print(grid_str)

    # Build resulting figure
    fig = go.Figure(layout=layout)

    # Attach subpot grid info to the figure
    fig.__dict__['_grid_ref'] = grid_ref
    fig.__dict__['_grid_str'] = grid_str

    return fig


def _configure_shared_axes(layout, grid_ref, x_or_y, shared, row_dir):
    rows = len(grid_ref)
    cols = len(grid_ref[0])

    layout_key_ind = ['x', 'y'].index(x_or_y)

    if row_dir < 0:
        rows_iter = range(rows-1, -1, -1)
    else:
        rows_iter = range(rows)

    def update_axis_matches(first_axis_id, ref, remove_label):
        if ref is None:
            return first_axis_id

        colspan = ref['spec']['colspan']

        if ref['subplot_type'] == '2d' and colspan == 1:
            if first_axis_id is None:
                first_axis_name = ref['layout_keys'][layout_key_ind]
                first_axis_id = first_axis_name.replace('axis', '')
            else:
                axis_name = ref['layout_keys'][layout_key_ind]
                axis_to_match = layout[axis_name]
                axis_to_match.matches = first_axis_id
                if remove_label:
                    axis_to_match.showticklabels = False

        return first_axis_id

    if shared == 'columns' or (x_or_y == 'x' and shared is True):
        for c in range(cols):
            first_axis_id = None
            for r in rows_iter:
                ref = grid_ref[r][c]
                first_axis_id = update_axis_matches(first_axis_id, ref, True)

    elif shared == 'rows' or (x_or_y == 'y' and shared is True):
        for r in rows_iter:
            first_axis_id = None
            for c in range(cols):
                ref = grid_ref[r][c]
                first_axis_id = update_axis_matches(first_axis_id, ref, True)
    elif shared == 'all':
        first_axis_id = None
        for c in range(cols):
            for r in rows_iter:
                ref = grid_ref[r][c]

                if x_or_y == 'y':
                    ok_to_remove_label = c > 0
                else:
                    ok_to_remove_label = r > 0 if row_dir > 0 else r < rows - 1

                first_axis_id = update_axis_matches(first_axis_id, ref, ok_to_remove_label)


def _init_subplot_2d(
        layout, x_domain, y_domain, max_subplot_ids=None
):
    if max_subplot_ids is None:
        max_subplot_ids = _get_initial_max_subplot_ids()

    # Get axis label and anchor
    x_cnt = max_subplot_ids['xaxis'] + 1
    y_cnt = max_subplot_ids['yaxis'] + 1

    # Compute x/y labels (the values of trace.xaxis/trace.yaxis
    x_label = "x{cnt}".format(cnt=x_cnt)
    y_label = "y{cnt}".format(cnt=y_cnt)

    # Anchor x and y axes to each other
    x_anchor, y_anchor = y_label, x_label

    # Build layout.xaxis/layout.yaxis containers
    xaxis_name = 'xaxis{cnt}'.format(cnt=x_cnt)
    yaxis_name = 'yaxis{cnt}'.format(cnt=y_cnt)
    x_axis = {'domain': x_domain, 'anchor': x_anchor}
    y_axis = {'domain': y_domain, 'anchor': y_anchor}

    layout[xaxis_name] = x_axis
    layout[yaxis_name] = y_axis

    ref_element = {
        'subplot_type': '2d',
        'layout_keys': (xaxis_name, yaxis_name),
        'trace_kwargs': {'xaxis': x_label, 'yaxis': y_label}
    }

    # increment max_subplot_ids
    max_subplot_ids['xaxis'] = x_cnt
    max_subplot_ids['yaxis'] = y_cnt

    return ref_element


def _init_subplot_single(
        layout, subplot_type, x_domain, y_domain, max_subplot_ids=None
):
    if max_subplot_ids is None:
        max_subplot_ids = _get_initial_max_subplot_ids()

    # Add scene to layout
    cnt = max_subplot_ids[subplot_type] + 1
    label = '{subplot_type}{cnt}'.format(subplot_type=subplot_type, cnt=cnt)
    scene = dict(domain={'x': x_domain, 'y': y_domain})
    layout[label] = scene

    trace_key = ('subplot'
                 if subplot_type in _subplot_prop_named_subplot
                 else subplot_type)

    ref_element = {
        'subplot_type': subplot_type,
        'layout_keys': (label,),
        'trace_kwargs': {trace_key: label}}

    # increment max_subplot_id
    max_subplot_ids['scene'] = cnt

    return ref_element


def _init_subplot_domain(x_domain, y_domain):
    # No change to layout since domain traces are labeled individually
    ref_element = {
        'subplot_type': 'domain',
        'layout_keys': (),
        'trace_kwargs': {'domain': {'x': x_domain, 'y': y_domain}}}

    return ref_element


def _init_subplot(
        layout, subplot_type, x_domain, y_domain, max_subplot_ids=None
):
    if max_subplot_ids is None:
        max_subplot_ids = _get_initial_max_subplot_ids()

    # Clamp domain elements between [0, 1].
    # This is only needed to combat numerical precision errors
    # See GH1031
    x_domain = [max(0.0, x_domain[0]), min(1.0, x_domain[1])]
    y_domain = [max(0.0, y_domain[0]), min(1.0, y_domain[1])]

    if subplot_type == '2d':
        ref_element = _init_subplot_2d(
            layout, x_domain, y_domain, max_subplot_ids
        )
    elif subplot_type in _subplot_types:
        ref_element = _init_subplot_single(
            layout, subplot_type, x_domain, y_domain, max_subplot_ids
        )
    elif subplot_type == 'domain':
        ref_element = _init_subplot_domain(x_domain, y_domain)
    else:
        raise ValueError('Invalid subplot type {subplot_type}'
                         .format(subplot_type=subplot_type))

    return ref_element


def _get_cartesian_label(x_or_y, r, c, cnt):
    # Default label (given strictly by cnt)
    label = "{x_or_y}{cnt}".format(x_or_y=x_or_y, cnt=cnt)
    return label


def _build_subplot_title_annotations(subplot_titles, list_of_domains, title_edge='top'):
    # If shared_axes is False (default) use list_of_domains
    # This is used for insets and irregular layouts
    # if not shared_xaxes and not shared_yaxes:
    x_dom = list_of_domains[::2]
    y_dom = list_of_domains[1::2]
    subtitle_pos_x = []
    subtitle_pos_y = []

    if title_edge == 'top':
        text_angle = 0
        xanchor = 'center'
        yanchor = 'bottom'

        for x_domains in x_dom:
            subtitle_pos_x.append(sum(x_domains) / 2.0)
        for y_domains in y_dom:
            subtitle_pos_y.append(y_domains[1])
    elif title_edge == 'right':
        text_angle = 90
        xanchor = 'left'
        yanchor = 'middle'

        for x_domains in x_dom:
            subtitle_pos_x.append(x_domains[1])
        for y_domains in y_dom:
            subtitle_pos_y.append(sum(y_domains) / 2.0)
    else:
        raise ValueError("Invalid annotation edge '{edge}'"
                         .format(edge=title_edge))

    plot_titles = []
    for index in range(len(subplot_titles)):
        if not subplot_titles[index] or index >= len(subtitle_pos_y):
            pass
        else:
            plot_titles.append({'y': subtitle_pos_y[index],
                                'xref': 'paper',
                                'x': subtitle_pos_x[index],
                                'yref': 'paper',
                                'text': subplot_titles[index],
                                'showarrow': False,
                                'font': dict(size=16),
                                'xanchor': xanchor,
                                'yanchor': yanchor,
                                'textangle': text_angle,
                                })
    return plot_titles


def _build_grid_str(specs, grid_ref, insets, insets_ref, row_seq):

    # Compute rows and columns
    rows = len(specs)
    cols = len(specs[0])

    # Initialize constants
    sp = "  "  # space between cell
    s_str = "[ "  # cell start string
    e_str = " ]"  # cell end string
    colspan_str = '       -'  # colspan string
    rowspan_str = '       :'  # rowspan string
    empty_str = '    (empty) '  # empty cell string
    # Init grid_str with intro message
    grid_str = "This is the format of your plot grid:\n"

    # Init tmp list of lists of strings (sorta like 'grid_ref' but w/ strings)
    _tmp = [['' for c in range(cols)] for r in range(rows)]

    # Define cell string as function of (r, c) and grid_ref
    def _get_cell_str(r, c, ref):
        ref_str = ','.join(ref['layout_keys'])
        return '({r},{c}) {ref}'.format(
            r=r + 1,
            c=c + 1,
            ref=ref_str)

    # Find max len of _cell_str, add define a padding function
    cell_len = max([len(_get_cell_str(r, c, ref))
                    for r, row_ref in enumerate(grid_ref)
                    for c, ref in enumerate(row_ref)
                    if ref]) + len(s_str) + len(e_str)

    def _pad(s, cell_len=cell_len):
        return ' ' * (cell_len - len(s))

    # Loop through specs, fill in _tmp
    for r, spec_row in enumerate(specs):
        for c, spec in enumerate(spec_row):

            ref = grid_ref[r][c]
            if ref is None:
                if _tmp[r][c] == '':
                    _tmp[r][c] = empty_str + _pad(empty_str)
                continue

            if spec['rowspan'] > 1:
                cell_str = '⎡' + _get_cell_str(r, c, ref)
            else:
                cell_str = s_str + _get_cell_str(r, c, ref)

            if spec['colspan'] > 1:
                for cc in range(1, spec['colspan'] - 1):
                    _tmp[r][c + cc] = colspan_str + _pad(colspan_str)

                if spec['rowspan'] > 1:
                    _tmp[r][c + spec['colspan'] - 1] = \
                        (colspan_str + _pad(colspan_str + e_str)) + ' ⎤'
                else:
                    _tmp[r][c + spec['colspan'] - 1] = \
                        (colspan_str + _pad(colspan_str + e_str)) + e_str
            else:
                cell_str += e_str

            if spec['rowspan'] > 1:
                for rr in range(1, spec['rowspan'] - 1):
                    _tmp[r + rr][c] = rowspan_str + _pad(rowspan_str)
                for cc in range(spec['colspan']):
                    _tmp[r + spec['rowspan'] - 1][c + cc] = (
                            rowspan_str + _pad(rowspan_str))

            _tmp[r][c] = cell_str + _pad(cell_str)

    # Append grid_str using data from _tmp in the correct order
    for r in row_seq[::-1]:
        grid_str += sp.join(_tmp[r]) + '\n'

    # Append grid_str to include insets info
    if insets:
        grid_str += "\nWith insets:\n"
        for i_inset, inset in enumerate(insets):
            r = inset['cell'][0] - 1
            c = inset['cell'][1] - 1
            ref = grid_ref[r][c]

            grid_str += (
                    s_str + ','.join(insets_ref[i_inset]['layout_keys']) + e_str +
                    ' over ' +
                    s_str + _get_cell_str(r, c, ref) + e_str + '\n'
            )
    return grid_str


def _set_trace_grid_reference(trace, layout, grid_ref, row, col):
    if row <= 0:
        raise Exception("Row value is out of range. "
                        "Note: the starting cell is (1, 1)")
    if col <= 0:
        raise Exception("Col value is out of range. "
                        "Note: the starting cell is (1, 1)")
    try:
        ref = grid_ref[row - 1][col - 1]
    except IndexError:
        raise Exception("The (row, col) pair sent is out of "
                        "range. Use Figure.print_grid to view the "
                        "subplot grid. ")

    trace_type = trace['type']
    trace_subplot_type = _trace_to_subplot.get(trace_type, None)

    # Validate that this trace is compatible with the subplot type

    # Update trace reference
    trace.update(ref['trace_kwargs'])

    # TODO: add validation. check that `ref['layout_keys']` are
    #  present in layout
    #
    # If not, raise informative error message about the incompatibility

    # if 'scene' in ref:
    #     trace.update(ref)
    #
    #     if ref['scene'] not in layout:
    #         raise Exception("Something went wrong. "
    #                         "The scene object for ({r},{c}) "
    #                         "subplot cell "
    #                         "got deleted.".format(r=row, c=col))
    # else:
    #     trace.update(ref)
    #
    #     xaxis_key = "xaxis{ref}".format(ref=ref['xaxis'][1:])
    #     yaxis_key = "yaxis{ref}".format(ref=ref['xaxis'][1:])
    #     if (xaxis_key not in layout
    #             or yaxis_key not in layout):
    #         raise Exception("Something went wrong. "
    #                         "An axis object for ({r},{c}) subplot "
    #                         "cell got deleted.".format(r=row, c=col))
