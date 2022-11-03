# encoding=utf-8
"""
gs_Plot_Tweets creates charts and maps to visualize social media datasets like Tweets.
galindosoft by Brian G. Herbert

"""
from math import fabs, pow
import datetime as dt
import copy
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from gs_data_dictionary import GSC, TRACE_COLRS
from gs_data_dictionary import RAWDIR
from plotly.subplots import make_subplots

plt_cfg = {"displayModeBar": False, "showTips": False}
pio.renderers.default = 'browser'
pio.templates.default = "plotly"
pd.options.plotting.backend = "plotly"
pd.options.display.precision = 3
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 12)

def get_employee_timeuse(emp_time, fp: str=RAWDIR):
    """
    get department of labor American worker time use data
    Series, Year, Period, Label, Value
    :param emp_time:
    :param fp: path to file, default is ./rawdata
    :return: pd.DataFrame
    """
    return pd.read_csv(fp + emp_time)

def get_bls_data(bls_f, fp: str=RAWDIR):
    """
    reads bureau of labor statistics worker data (csv file)
    :param bls_f: str 'bls_labor_bysector.csv' or similar
    :param fp: path for raw input files, default=./rawdata
    :return: pd.DataFrame
    """
    return pd.read_csv(fp + bls_f, dtype={'1Q2022': float, '2Q2022': float})

def do_histo(twl, col: str=None):
    """
    produces plotly histogram for one column or var
    :param twl: list of values
    :param col: name of column to plot
    :return:
    """
    import plotly.express as px
    fig = px.histogram(twl, x=col)
    fig.show()

    return

def convert_cloud_to_plotly(mpl_cld):
    """
    converts a matplotlib based word cloud to plotly figure
    :param mpl_cld: the mpl based wordcloud generated in filter_and_enrich.py
    :return: plotly figure for wordcloud
    """
    from plotly.tools import mpl_to_plotly

    return mpl_to_plotly(mpl_cld)

def do_sent_classify(df: pd.DataFrame, clrcol: str = "compound"):
    """
    create classifications based on a sentiment score type
    :param df:
    :return: pd.DataFrame
    """
    if clrcol in df.columns:
        s_dev = round(float(df[clrcol].std()), ndigits=1)
        s_med = round(float(df[clrcol].median()), ndigits=1)
        upper = s_med + s_dev
        lower = s_med - s_dev

        def do_class(sntx: float):
            if sntx > upper:
                return "rgb(0, 102, 0)"
            elif sntx > lower:
                return "rgb(204, 204, 153)"
            else:
                return "rgb(255, 51, 153)"

        df['snt_clr'] = df[clrcol].apply(lambda x: do_class(x))

        return df
    else:
        print("Error applying sentiment classification")
        return None

def create_layout():
    """
    plotly uses set of dictionaries to define layout for a graph_objects plot.
    this function allows a layout instance to be shared across plots in this app
        once instantiated, object properties can be set directly 'xaxis.title=',
        or creating/modifying objects via plotly Fxs like 'update_layout
    working towards standard typeface, sizes, colors, etc in my apps, such as:
        Helvetica Neue Thin for text, and Copperplate for legends

    :return: plotly layout
    """
    gs_lyt = go.Layout(height=850, width=1600,
                       title={'font': {'size': 32, 'family': 'Helvetica Neue Medium',
                                       'color': GSC['oblk']
                                       }
                              },
                       paper_bgcolor=GSC['ltgry'],
                       font={'size': 18, 'family': 'Helvetica Neue Light'},
                       hovermode="closest",
                       hoverdistance=10,
                       showlegend=True,
                       legend={'title': {'font': {'size': 20, 'family': 'Copperplate Light'}},
                               'font': {'size': 18, 'family': 'Copperplate Light', 'color': GSC["dkryl"]},
                               'bgcolor': GSC['beig'], 'bordercolor': GSC['oblk'],
                               'borderwidth': 2, 'itemsizing': "trace"
                               },
                       xaxis={'title': {'font': {'size': 24, 'family': 'Helvetica Neue UltraLight'}},
                              'linecolor': GSC['oblk'], 'rangemode': "normal",
                              'showspikes': True, 'spikethickness': 1,
                              },
                       yaxis={'title': {'font': {'size': 24, 'family': 'Helvetica Neue UltraLight'}},
                              'linecolor': GSC['oblk'],
                              'showspikes': True, 'spikethickness': 1,
                              },
                       margin=go.layout.Margin(autoexpand=True)
                       )
    gs_lyt.template.data.scatter = [
        go.Scatter(marker=dict(symbol="diamond", size=12)),
        go.Scatter(marker=dict(symbol="circle", size=12)),
        go.Scatter(marker=dict(symbol="triangle-up", size=12)),
        go.Scatter(marker=dict(symbol="square", size=12))
    ]
    gs_lyt.coloraxis.colorscale = [
        [0, '#0d0887'],
        [0.1111111111111111, '#46039f'],
        [0.2222222222222222, '#7201a8'],
        [0.3333333333333333, '#9c179e'],
        [0.4444444444444444, '#bd3786'],
        [0.5555555555555556, '#d8576b'],
        [0.6666666666666666, '#ed7953'],
        [0.7777777777777778, '#fb9f3a'],
        [0.8888888888888888, '#fdca26'],
        [1, '#f0f921']
    ]

    return gs_lyt

def show_figure(gofig: go.Figure):
    """
    -displays plotly figure object constructed in this module's functions.
    -control of config and renderer settings for Plotly's fig.show()
    - config of run-time preferences for visualization or save to image or disk.
    - default renderers: 'json', 'png', 'svg', 'chrome', 'browser', 'sphinx_gallery'

    -also: plotly.mimetype to render on display in an iPython context
    :param gofig: plotly.graph_objects.Figure object
    :return: None
    """
    print(pio.renderers)
    pio.renderers.default = 'chrome+browser+png+svg'
    # pio.renderers.keys()
    # if following is set, will show plots on display in python
    # pio.renderers.render_on_display = True

    gs_rend = pio.renderers.default

    cfg: dict = {"displayModeBar": False, "showTips": False}

    gofig.show(renderer=gs_rend, config=cfg)

    return gofig

def hist_quot_rply_rtwt(df: pd.DataFrame, plyt: go.Layout = None, appd: str = "Work Tweets"):
    """
    plot influence metrics as bar histogram
    :param df: tweet dataframe with metrics columns
    :param plyt: go.Layout
    :param appd: name of project or dataset
    :return:
    """
    bin_ct = 16
    total_tws= len(df)
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    lay.title.text = "PRJX- Tweet Influence Distribution<br>Value Transform Ltd.".replace("PRJX", appd)
    lay.xaxis.title.text = "Log of Tweet Count"
    lay.yaxis.title.text = "Log of Influence Count (Quotes, Retweets, and Replies)"
    lay.legend.title = f"Influence type for {total_tws} tweets"
    lay.yaxis.type = "linear"
    lay.xaxis.type = "log"
    lay.boxmode = "group"
    fig = go.Figure(layout=lay)
    fig.add_trace(go.Histogram(y=df['quot_log'],
                               nbinsy=bin_ct,
                               name="quotes",
                               marker_color=GSC["brnz"],
                               opacity=0.8,

                               ))
    fig.add_trace(go.Histogram(y=df['rtwt_log'],
                               nbinsy=bin_ct,
                               name="retweets",
                               marker_color=GSC["mgnta"],
                               opacity=0.7,

                               ))
    fig.add_trace(go.Histogram(y=df['rply_log'],
                               nbinsy=bin_ct,
                               name="replies",
                               marker_color=GSC["brwn"],
                               opacity=0.8,

                               ))
    fig.add_trace(go.Histogram(y=df['fave_log'],
                               nbinsy=bin_ct,
                               name="likes",
                               marker_color=GSC["dkblu"],
                               opacity=0.9,
                               ))

    # fig.update_layout(bargap=0.1)
    fig.update_traces()
    fig.show(config=plt_cfg)

    return fig

def show_sent_distribution(twdf, plyt: go.Layout, appd: str="Work Tweets"):
    """
    show box plots for sentiment scores of tweets
    :param twdf: pd.Dataframe with all score info
    :param plyt: plotly go.Layout object instance
    :param appd: str with domain or project name
    :return: None
    """
    tw_total: str = str(len(twdf))
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    lay.title.text = "Sentiment for TOT tweets on PRJX".replace("PRJX", appd).replace("TOT", tw_total)
    lay.xaxis.title.text = "Quartiles, Mean and St. Deviation (dashes), plus actuals"
    lay.yaxis.title.text = "Score Type"
    lay.legend.title = "Vader sentiment scores"

    fig = go.Figure(layout=lay)

    fig.add_trace(go.Box(x=twdf['neg'], quartilemethod="inclusive",
                         name="negative", marker_color=GSC['drkrd'],
                         boxmean='sd'))
    fig.add_trace(go.Box(x=twdf['pos'], quartilemethod="inclusive",
                         name="positive", marker_color=GSC['grn'],
                         boxmean='sd'))
    fig.add_trace(go.Box(x=twdf['compound'], quartilemethod="inclusive",
                         name="compound", marker_color=GSC['dkryl'],
                         boxmean='sd'))
    fig.add_trace(go.Box(x=twdf['neu'], quartilemethod="inclusive",
                         name="neutral", marker_color=GSC['brgry'],
                         boxmean='sd'))
    fig.update_traces(boxpoints='all', jitter=0.3)
    fig.show(config=plt_cfg)

    return fig

def show_sent_fromlist(twl, plyt: go.Layout, appd: str="Work Tweets"):
    """
    show box plots for sentiment scores of tweets
    :param twl: list of dict of tweets
    :param plyt: plotly go.Layout object instance
    :param appd: str with domain or project name
    :return: None
    """
    tw_total: str = str(len(twl))
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    lay.title.text = "Sentiment for TOT tweets on PRJX".replace("PRJX", appd).replace("TOT", tw_total)
    lay.xaxis.title.text = "Quartiles, Mean and St. Deviation (dashes), plus actuals"
    lay.yaxis.title.text = "Score Type"
    lay.legend.title = "Vader sentiment scores"
    compx = [x['compound'] for x in twl]
    posx = [x['pos'] for x in twl]
    negx = [x['neg'] for x in twl]
    neux = [x['neu'] for x in twl]

    fig = go.Figure(layout=lay)

    fig.add_trace(go.Box(x=negx, quartilemethod="inclusive",
                         name="negative", marker_color=GSC['drkrd'],
                         boxmean='sd'))
    fig.add_trace(go.Box(x=posx, quartilemethod="inclusive",
                         name="positive", marker_color=GSC['grn'],
                         boxmean='sd'))
    fig.add_trace(go.Box(x=compx, quartilemethod="inclusive",
                         name="compound", marker_color=GSC['dkryl'],
                         boxmean='sd'))
    fig.add_trace(go.Box(x=neux, quartilemethod="inclusive",
                         name="neutral", marker_color=GSC['brgry'],
                         boxmean='sd'))
    fig.update_traces(boxpoints='all', jitter=0.3)
    fig.show(config=plt_cfg)

    return fig

def bar_hashtags(hashes: dict, plyt: go.Layout = None, stops: list= None, appd: str=""):
    """
    most frequent hashtags plotted with bar chart
    first- use utility in gs_tweet_analysis to sort descending plus filter out stoplists

    :param hashes: dict of hashtags with counts of occurrences
    :param mentions: dict of user mentions with counts of occurrences
    :param plyt: plotly layout instance, creates a copy so not to munge shared elements
    :param stops: listof words to exclude from cloud
    :param appd: str with domain or project name
    :return:
    """
    hsh_limit: int = 21
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()
    lay.title.text = "PRJX- Top Hashtags in Dataset<br>a ValueTransform Ltd. project".replace("PRJX", appd)
    lay.title.font = {'size': 32, 'family': 'Helvetica Neue Medium', 'color': GSC['oblk']}
    lay.xaxis.tickangle = -60
    lay.xaxis.tickfont = {'size': 24, 'family': 'Copperplate Light'}
    lay.yaxis.title = 'Count of Hashtags in Dataset'
    lay.yaxis.title.font = {'size': 28, 'family': 'Copperplate Light'}
    lay.showlegend = False
    # lay.legend.itemsizing = 'constant'
    # lay.legend.title = "Work-Related tags"
    lay.margin.b = 160
    fig = go.Figure(layout=lay)

    for k in stops:
        if k in hashes:
            hashes.pop(k)

    srt: list = sorted(hashes, key=lambda x: int(hashes[x]), reverse=True)
    plot_hash: dict = {k: hashes[k] for k in srt}
    h_x: list = []
    h_y: list = []
    h_c: list = []
    for h, ct in zip(plot_hash.items(), range(hsh_limit)):
        h_x.append(h[0])
        h_y.append(h[1])
        h_c.append(TRACE_COLRS[ct])

    fig.add_trace(go.Bar(name="Top Hashtags", x=h_x, y=h_y, text=h_x,
                         marker=dict(line_width=2, color=h_c),
                         texttemplate="%{x}<br>count: %{y}",
                         textangle=-90, textfont=dict(size=20)
                         ))

    fig.show(config=plt_cfg)

    return fig

def bar_tags_horizontal(hashes: dict, plyt: go.Layout = None, stops: list= None, appd: str=""):
    """
    most frequent hashtags plotted across the page
    :param hashes: dict of hashtags with counts of occurrences
    :param mentions: dict of user mentions with counts of occurrences
    :param plyt: plotly layout instance, creates a copy so not to munge shared elements
    :param stops: listof words to exclude from cloud
    :param appd: str with domain or project name
    :return:
    """
    hsh_limit: int = 20
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()
    lay.title.text = "PRJX- Top Hashtags in Dataset<br>a ValueTransform Ltd. project".replace("PRJX", appd)
    lay.title.font = {'size': 32, 'family': 'Helvetica Neue Medium', 'color': GSC['oblk']}
    # lay.xaxis.tickangle = -60
    # lay.yaxis.tickfont = {'size': 24, 'family': 'Copperplate Light'}
    lay.yaxis.showticklabels = False
    lay.xaxis.title = 'Count of Hashtags in Dataset'
    lay.xaxis.title.font = {'size': 28, 'family': 'Copperplate Light'}
    lay.showlegend = False
    # lay.legend.itemsizing = 'constant'
    # lay.legend.title = "Work-Related tags"
    lay.margin.l = 10
    lay.margin.r = 10
    lay.margin.b = 40
    fig = go.Figure(layout=lay)

    for k in stops:
        if k in hashes:
            hashes.pop(k)

    srt = sorted(hashes, key=lambda x: int(hashes[x]), reverse=True)
    plot_hash: dict = {k: hashes[k] for k in srt}
    h_x: list = []
    h_y: list = []
    h_c: list = []
    for h, ct in zip(plot_hash.items(), range(hsh_limit)):
        # need to reverse order of insertion so top tag shows at top of horizontal
        h_x.insert(0, h[0])
        h_y.insert(0, h[1])
        h_c.insert(0, TRACE_COLRS[ct])

    # can insert this if labels look off: textposition="inside"
    fig.add_trace(go.Bar(name="Top Hashtags", x=h_y, y=h_x, text=h_x,
                         marker=dict(line_width=2, color=h_c),
                         texttemplate="%{y} (%{x})",
                         textangle=0, orientation="h",
                         insidetextanchor="start",
                         insidetextfont=dict(family='Copperplate Regular',size=24,color=GSC['owht']),
                         outsidetextfont=dict(family='Copperplate Regular',size=24,color=GSC['oblk']),
                         ))
    fig.show(config=plt_cfg)

    return fig

def labor_scatter(blsdf: pd.DataFrame, plyt: go.Layout =None, appd: str=None):
    """
    plotly 3d chart with use of custom layout template
    :param blsdf: pd.DataFrame of public events
    :param plyt: plotly go.Layout object instance
    :return:
    """
    SYMBOL_TYP: list = ["diamond", "circle", "triangle-up", "diamond", "circle"]
    lay: go.Layout = copy.copy(plyt) if plyt else create_layout()

    cols: list = list(blsdf.columns)
    cols.remove('Measure')
    lay.title.text = "PRJX- US Worker Employment and Productivity<br>a ValueTransform Ltd. project".replace("PRJX",
                                                                                                            appd)
    # lay.xaxis.autorange = False
    lay.xaxis.type = "category"
    lay.xaxis.range = [0, len(cols)]
    lay.xaxis.showticklabels = True
    lay.xaxis.tickangle = -45
    lay.margin.b = 80
    lay.legend.title = "BLS US Worker Measure"
    lay.xaxis.title = "By Quarter, 2012 through 2Q 2022"
    lay.yaxis.title = "All Measures Indexed to 2012=100"
    lay.xaxis.title.font = {'size': 28, 'family': 'Copperplate Light'}
    lay.yaxis.title.font = {'size': 28, 'family': 'Helvetica Neue UltraLight'}
    lay.xaxis.tickfont = {'size': 14, 'family': 'Copperplate Light'}

    fig = go.Figure(layout=lay)
    fig.layout.xaxis.titlefont = dict(size=20, color=GSC['oblk'])
    fig.update_layout(lay)

    pltdct = blsdf.to_dict("records")
    for x in range(len(pltdct)):
        tracex: list = []
        tracey: list = []
        for k, v in pltdct[x].items():
            if str(k).startswith('Measure'):
                tracename: str = v
            else:
                tracex.append(k)
                tracey.append(v)
        fig.add_trace(go.Scatter(x=tracex, y=tracey,
                                 mode='markers',
                                 name=tracename,
                                 visible=True,
                                 marker=dict(symbol=SYMBOL_TYP[x], size=10,
                                             color=TRACE_COLRS[x],
                                             )
                                 ),
                      )

    fig.show(config=plt_cfg)

    return fig

def gain_clr(varx: float, vmean: float=3.4, vstd: float=25):
    """
    fx to customize color based on attribute value
    :param varx: field from dataframe
    :param vmean: avg for the field
    :param vstd: std deviation for field
    :return: rgb color
    """

    if varx > (vmean + vstd):
        return GSC["dkgrn"]
        # return 0.9
    elif varx > 0:
        return GSC["gray"]
        # return 0.6
    elif varx > (vmean - vstd):
        return GSC["drkrd"]
        # return 0.3
    else:
        return GSC['mgnta']
        # return 0.0

def plot_atudata(timedf: pd.DataFrame, plyt: go.Layout, appd: str = None):
    """
    uses plotly dual y-axis to plot both market data and tweet data
    :param timedf: pd.DataFrame with Tweets
    :param plyt: plotly go.Layout object instance
    :param appd: str of project or domain name
    :return: None
    """
    SYMBOL_TYP: list = ["diamond", "circle", "triangle-up", "diamond", "circle"]
    lay: go.Layout = copy.copy(plyt) if plyt else create_layout()

    lay.margin = {'l': 50, 'r': 80, 't': 50, 'b': 50}
    lay.legend.title = "Hours/Day spent on:"
    lay.legend.itemsizing = "constant"
    lay.title.text = "PRJX - US Adult Survey Series - Use of Time".replace("PRJX", appd)
    lay.xaxis.title = "Year"
    lay.yaxis.title = "Hours / Day"
    fig = go.Figure(layout=lay)

    tracename: list = list(timedf.Series.unique())
    tracetype_cnt: int = len(tracename)
    for tr in range(tracetype_cnt):
        tmp: dict = timedf.loc[timedf.Series == tracename[tr], ['Year', 'Value']].to_dict("list")
        year_cnt: int = len(tmp['Year'])
        tracex: list = []
        tracey: list = []
        # diff shows difference from prior period on hover
        diff: list = [0]
        meas_type: str = tracename[tr]
        for x in range(year_cnt):
            tracex.append(tmp['Year'][x])
            tracey.append(tmp['Value'][x])
            if x > 0:
                diff.append(tracey[x] - tracey[x-1])

        fig.add_trace(go.Scatter(
            x=tracex,
            y=tracey,
            name=meas_type,
            mode='lines+markers',
            line=dict(dash='dash'),
            textposition="bottom center",
            hovertemplate="<b>Year: %{x}</b>" +
                          "<br>hours/day: %{y:.2f}" +
                          "<br>gain/loss: %{customdata[i]: .2f}",
            hoverlabel={'font': {'family': 'Copperplate Light', 'size': 14}},
            marker=dict(color=TRACE_COLRS[tr*2], opacity=0.8, size=14,
                        symbol=SYMBOL_TYP[tr],
                        ),
            customdata=diff,
        ))

    fig.show(config=plt_cfg)

    return fig

def plot_blsdata(labordf: pd.DataFrame, plyt: go.Layout, appd: str = None):
    """
    uses plotly dual y-axis to plot both market data and tweet data
    :param labordf: pd.DataFrame with productivity data
    :param plyt: plotly go.Layout object instance
    :param appd: str of project or domain name
    :return: None
    """
    SYMBOL_TYP: list = ["diamond", "circle", "triangle-up", "diamond", "circle"]
    lay: go.Layout = copy.copy(plyt) if plyt else create_layout()
    lay.height = 900
    lay.margin = {'l': 60, 'r': 80, 't': 70, 'b': 100}
    lay.legend.title = "Type of Data"
    lay.legend.itemsizing = "constant"
    lay.title.text = "PRJX - Bureau of Labor Statistics Worker Data".replace("PRJX", appd)
    lay.xaxis.title = "Quarter and Year"
    fig = go.Figure(layout=lay)

    pltdct = labordf.to_dict("records")
    for x in range(len(pltdct)):
        tracex: list = []
        tracey: list = []
        for k, v in pltdct[x].items():
            if str(k).startswith('Measure'):
                tracename: str = v
            else:
                tracex.append(k)
                tracey.append(v)
        fig.add_trace(go.Scatter(x=tracex, y=tracey,
                                 name=tracename,
                                 visible=True, mode="lines+markers",
                                 marker=dict(color=TRACE_COLRS[x], symbol=SYMBOL_TYP[x], size=12)
                                 ),
                      )

    fig.update_layout(lay, overwrite=False)
    fig.show(config=plt_cfg)

    return fig

def plot_3d_scatter(twdf, plyt: go.Layout, appd: str="Work Tweets"):
    """
    plot_3d_scatter uses 3 dimensions plus size and color to graphically represent tweets
    :param twdf: pd.DataFrame with normalized (scaled) features
    :param plyt: plotly go.Layout object instance
    :param appd: str with name of domain or project
    :return: None
    """
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    lay.title.text = "Influential Tweets, scaled size <b>color coded sentiment</b>"
    lay.scene = {'xaxis': {'title': 'Sorted by Date', 'spikethickness': 1},
                   'yaxis': {'title': 'quote-retweet-reply scaled count', 'spikethickness': 1},
                   'zaxis': {'title': 'favorite(like) scaled count', 'spikethickness': 1},
                   'aspectmode': 'manual',
                   'aspectratio': {'x': 2, 'y': 2, 'z': 2}
                   }
    lay.xaxis.title.standoff = 10
    lay.yaxis.title.standoff = 10
    lay.xaxis.automargin = True
    sizing: float = round(twdf['inflcode'].max()/24, ndigits=1)
    pltclr: list = twdf['compcode'].apply(lambda x: TRACE_COLRS[x])

    fig = go.Figure(data=go.Scatter3d(
        x=twdf['compound_rsc'],
        y=twdf['qrr_zsc'],
        z=twdf['fave_zsc'],
        hovertemplate='Tweet sent on: %{meta}<br>' +
                      'x:<i><b>compound sentiment</b></i>: %{x}' +
                      '<br>y:<b>QRR count</b>: %{y:.2f}, ' +
                      'z:<b>Favorite</b>: %{z:.2f}' +
                      "<br>Influence (size code): %{customdata}" +
                      "<br>%{text}",
        text=twdf['text'],
        name=appd,
        mode='markers',
        showlegend=False,
        marker=dict(
            sizemode='diameter',
            sizeref=sizing,
            sizemin=6,
            size=twdf['inflcode'],
            opacity=0.8,
            color=pltclr,
        ),
        customdata=twdf['inflcode'],
        meta=twdf['sent'].apply(lambda x: dt.datetime.strftime(x, "%b %d, %Y"))
    ), layout=lay
    )
    fig.update_layout(lay)
    fig.show(config=plt_cfg)

    return fig

def plot3d_bydate(twdf: pd.DataFrame, plyt: go.Layout, appd: str="Work Tweets"):
    """
    plot_3d_scatter uses 3 dimensions plus size and color to graphically represent tweets
    :param twdf: pd.DataFrame with normalized (scaled) features
    :param plyt: plotly go.Layout object instance
    :param appd: str with name of domain or project
    :return: None
    """
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    plotdf: pd.DataFrame = twdf.copy()

    lay.title.text = "PRJX -Influence and Sentiment <br>sentiment-color and distribution-size".replace("PRJX", appd)
    lay.scene = {'xaxis': {'title': 'Sorted by Date', 'spikethickness': 1},
                   'yaxis': {'title': 'quote-retweet-reply scaled count', 'spikethickness': 1},
                   'zaxis': {'title': 'favorite(like) scaled count', 'spikethickness': 1},
                   'aspectmode': 'data',
                   }
    # if needed in lay.scene:  'aspectratio': {'x': 3, 'y': 2, 'z': 2}
    lay.xaxis.title.standoff = 10
    lay.yaxis.title.standoff = 10
    lay.xaxis.automargin = True
    lay.yaxis.autorange= True

    sizing: float = round(twdf['inflcode'].max()/28, ndigits=1)
    pltclr: list = twdf['compcode'].apply(lambda x: TRACE_COLRS[x])

    fig = go.Figure(data=go.Scatter3d(
        x=twdf['tdate'],
        y=twdf['qrr_zsc'],
        z=twdf['fave_zsc'],
        hovertemplate='Tweet sent on: %{meta}<br>' +
                      'compound sentiment: %{customdata:.2f}' +
                      '<br><b>QRR z-score</b>: %{y:.2f}, ' +
                      '<b>Favorite z-score</b>: %{z:.2f}' +
                      "<br>%{text}",
        text=twdf['text'],
        name=appd,
        mode='markers',
        showlegend=False,
        marker=dict(
            sizemode='diameter',
            sizeref=sizing,
            sizemin=6,
            size=twdf['redist'],
            opacity=0.8,
            color=pltclr,
        ),
        customdata=twdf['compound'],
        meta=twdf['tdate'].apply(lambda x: dt.datetime.strftime(x, "%b %d, %Y"))
    ), layout=lay
    )

    fig.update_layout(lay)
    fig.show(config=plt_cfg)

    return fig

def plot_3d_from_list(twlst, plyt: go.Layout, appd: str="Work Tweets"):
    """
    plot_3d_scatter uses 3 dimensions plus size and color to graphically represent tweets
    :param twdf: pd.DataFrame with normalized (scaled) features
    :param plyt: plotly go.Layout object instance
    :param appd: str with name of domain or project
    :return: None
    """
    plx = [x['compound'] for x in twlst]
    ply = [x['qrr'] for x in twlst]
    plz = [x['fave'] for x in twlst]
    pldist = [x['redist'] for x in twlst]
    pltxt = [x['text'] for x in twlst]
    plcomp = [TRACE_COLRS[x['compcode']] for x in twlst]

    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    lay.title.text = "Influential Tweets, color coded sentiment<br>"
    lay.scene = {'xaxis': {'title': 'Compound Sentiment', 'spikethickness': 1},
                   'yaxis': {'title': 'qt-rt-rply count', 'spikethickness': 1},
                   'zaxis': {'title': 'likes count', 'spikethickness': 1},
                   'aspectratio': {'x': 2, 'y': 2, 'z': 2}
                   }
    lay.xaxis.automargin = True
    lay.yaxis.range = [0, max(ply)]

    fig = go.Figure(data=go.Scatter3d(
        x=plx,
        y=ply,
        z=plz,
        hovertemplate='x:<i><b>Compound sentiment</b></i>: %{x}' +
                      '<br>y:<b>Q-R-R</b>: %{y:.2f}' +
                      '<br>z:<b>Likes</b>: %{z:.2f}' +
                      '<br>text: %{text}',
        text=pltxt,
        name=appd,
        mode='markers',
        showlegend=False,
        marker=dict(
            sizemode='diameter',
            sizeref=max(pldist)/20,
            sizemin=8,
            size=(pldist * 8),
            opacity=0.9,
            color=plcomp,
        ),), layout=lay
    )

    fig.update_layout(lay)

    fig.show(config=plt_cfg)

    return fig

def scat3d_list_bydate(twlst, plyt: go.Layout, appd: str="Work Tweets"):
    """
    plot_3d_scatter uses 3 dimensions plus size and color to graphically represent tweets
    :param twdf: pd.DataFrame with normalized (scaled) features
    :param plyt: plotly go.Layout object instance
    :param appd: str with name of domain or project
    :return: None
    """
    plx = [x['tdate'] for x in twlst]
    ply = [x['qrr'] for x in twlst]
    plz = [x['fave'] for x in twlst]
    pldist = [x['redist'] for x in twlst]
    pltxt = [x['text'] for x in twlst]
    plsent = [x['compound'] for x in twlst]
    plcomp = [TRACE_COLRS[x['compcode']] for x in twlst]

    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    lay.title.text = "PRJX -Influence and Sentiment <br>sentiment-color and distribution-size".replace("PRJX", appd)
    lay.scene = {'xaxis': {'title': 'Date of Tweet', 'spikethickness': 1},
                   'yaxis': {'title': 'QT-RT-Reply (log)', 'spikethickness': 1},
                   'zaxis': {'title': 'Likes (log)', 'spikethickness': 1},
                   'aspectmode': 'data',
                   }
    lay.xaxis.automargin = True
    lay.yaxis.range = [0, max(ply)]

    fig = go.Figure(data=go.Scatter3d(
        x=plx,
        y=ply,
        z=plz,
        hovertemplate='x:<b>Date</b>: %{x}' +
                      '<br>y:<b>Q-R-R count</b>: %{y:.2f}' +
                      '<br>z:<b>Likes</b>: %{z:.2f}' +
                      '<br>compound sentiment (color coded): %{meta:.2f}' +
                      '<br>text: %{text}',
        text=pltxt,
        name=appd,
        mode='markers',
        showlegend=False,
        marker=dict(
            sizemode='diameter',
            sizeref=max(pldist)/20,
            sizemin=8,
            size=(pldist * 6),
            opacity=0.9,
            color=plcomp,
        ),meta=plsent,
    ), layout=lay
    )

    fig.update_layout(lay)
    fig.show(config=plt_cfg)

    return fig

def plot_3d(rdf: pd.DataFrame, plyt: go.Layout = None, appd: str=None, styp: str = "compound"):
    """
    rewrite of scatter 3d to plot 6-hour blocks and scale marker size by influence
    holding place for annotation code- in case I put it back in future:
        annot_rel = {
        'xref': 'paper', 'yref': 'paper',
        'x': 0.1, 'y': 0.0, 'xanchor': 'left', 'yanchor': 'bottom',
        'text': 'plot of influential tweets with sentimenty<br>',
        'bgcolor': "rgb(153, 153, 153)",
        'showarrow': False,
        'font': {'size': 14, 'color': "rgb(25, 25, 25)"}
    }
    lay.annotations = [annot_rel]

    :param rdf: pd.DataFrame with normalized features (use do_scaling function)
    :param plyt: plotly go.Layout object instance
    :param appd: str with domain or project name
    :param styp: str with sentiment field name to use for plot
    :return: None
    """
    twt: pd.DataFrame = rdf.copy(deep=True)
    siz_const: int = 15  # marker size: divide max value by this for sizeref
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    sntdct: dict = twt[styp].value_counts().to_dict()
    sntdct = {y: sntdct[y] for y in sorted(sntdct, key=lambda x: fabs(x), reverse=True)}
    sntsum = sum(sntdct.values())

    def set_color(metric: float, segs: int = 4):
        """
        passed a float field from a dataframe, returns an rgb color to use for marker
        :param metric: float
        :return: str in form "rbg(0, 0, 0)"
        """
        sntslice: int = round(sntsum / segs, ndigits=0)
        aggr: int = 0
        for k, v in sntdct.items():
            aggr += v
            if fabs(k) < fabs(metric):
                if aggr < sntslice:
                    # return "rgb(204, 51, 51)"
                    return 1.0
                elif (aggr > sntslice):
                    # return "rgb(153, 102, 51)"
                    return 0.7
                elif (aggr > 2 * sntslice):
                    # return "rgb(102, 153, 153)"
                    return 0.3
                else:

                    # return "rgb(102, 102, 102)"
                    return 0.0
        return

    def parse_lst(coly: list):
        """
        inner fx to parse a list in a column in a dataframe
        :param coly: a list of str
        return:
        """
        if isinstance(coly, list):
            if len(coly) > 0:
                # print(" %s is list with length > 0" % coly)
                lst_str: str = ""
                for hsh in coly:
                    hsh = str(hsh).lower()
                    tmp: str = " " + hsh
                    # print(" iter item is %s" % hsh)
                    lst_str: str = lst_str + tmp
                    lst_str = lst_str.strip()
                    # print("joined creation is %s" % lst_str)
                return lst_str
            else:
                # print(" length of hash list is 0")
                return None
        else:
            # print(" did not get list from hashtag field")
            return None

    def chk_typ(coly):
        """
        if Tweet is on all three lists (Q-R-R, Fave, and Sentiment) show differently
        :param coly:
        :return:
        """
        if coly in ['qfs', 'qs', 'fs']:
            return 3
        elif coly in ['qf']:
            return 2
        elif coly in ['q', 'f']:
            return 1
        else:
            return 0

    twt['s_color'] = twt[styp].apply(lambda x: set_color(x))
    clrlst: list = twt['s_color'].to_list()
    # aspectmode options are cube, manual, data, auto
    lay.xaxis.tickmode = "linear"
    lay.xaxis.dtick = 21600000
    lay.xaxis.tick0 = "May-01"
    lay.xaxis.tickformat = "%b-%d"
    lay.xaxis.tickfont = dict(size=10)
    lay.xaxis.showticklabels = False
    lay.title.text = "PRJX QT, RT, Reply and Like metrics<br>".replace("PRJX", appd) + \
                     "Color-coded Sentiment (light-pos, dark-neg), hover for details"
    lay.showlegend = False
    lay.scene = {'xaxis': {'title': "Date of Tweet", 'spikethickness': 1, 'dtick': 86400000,
                           'showtickprefix': None, 'tickformat': "%b-%d",
                           'type': 'date', 'tickfont': {'color': "rgb(51, 102, 153)",
                                                        'family': 'Helvetica Neue UltraLight',
                                                        'size': 10
                                                        }
                           },
                 'yaxis': {'title': 'Quotes-ReTweets-Replies<br>scaled', 'spikethickness': 1,
                           'showtickprefix': None,
                           'tickfont': {'color': "rgb(51, 102, 153)",
                                        'family': 'Helvetica Neue UltraLight', 'size': 14
                                        }
                           },
                 'zaxis': {'title': 'Likes - scaled', 'spikethickness': 1,
                           'tickfont': {'color': "rgb(51, 102, 153)",
                                        'family': 'Helvetica Neue UltraLight', 'size': 14
                                        }
                           },
                 'aspectmode': 'manual', 'aspectratio': {'x': 2, 'y': 1, 'z': 1},
                 }
    lay.margin.l = 100

    xlst: list = twt.sent.to_list()
    qrlst: list = list(twt['qrr_log'].apply(lambda x: "{:.2f}".format(x)))
    fvlst: list = list(twt['fave_log'].apply(lambda x: "{:.2f}".format(x)))
    sntlst: list = list(twt[styp].apply(lambda x: "{: .2f}".format(x)))
    txtlst: list = list(twt['hashes'].apply(lambda x: parse_lst(x)))

    mrglst: list = []
    for qm, fm, sm in zip(qrlst, fvlst, sntlst):
        mrglst.append((float(qm) + float(fm)) * pow(float(sm), 2))
    mrg_mean = sum([float(x) for x in mrglst]) / len(mrglst)
    mrg_max = max(mrglst)
    print("shared metric has mean of %.2f and max of %.2f " % (mrg_mean, mrg_max))

    fig = go.Figure(data=go.Scatter3d(x=xlst, y=qrlst, z=fvlst,
                                      hovertemplate='<b>Tweet on</b> %{x}' +
                                                    '<br>y: Scaled QRR-F: %{y:.2f}' +
                                                    "<br>Sentiment: %{customdata:.2f}" +
                                                    "<br>hashtags: %{text}",
                                      text=txtlst,
                                      name="Tweet Influence",
                                      mode='markers', showlegend=True,
                                      textsrc="%{customdata:.1f}",
                                      marker=dict(size=4,
                                                  opacity=0.8,
                                                  color=clrlst, colorscale='viridis'
                                                  ),
                                      customdata=sntlst
                                      ), layout=lay
                    )

    fig.update_layout(lay, margin=dict(l=30, r=50, b=30, t=60))
    fig.show(config=plt_cfg)

    return fig

def plot_3d_wrdscores(rdf: pd.DataFrame, plyt: go.Layout = None, appd: str=None, styp: str = "compound"):
    """
    rewrite of scatter 3d to plot 6-hour blocks and scale marker size by influence
    holding place for annotation code- in case I put it back in future:
        annot_rel = {
        'xref': 'paper', 'yref': 'paper',
        'x': 0.1, 'y': 0.0, 'xanchor': 'left', 'yanchor': 'bottom',
        'text': 'plot of influential tweets with sentimenty<br>',
        'bgcolor': "rgb(153, 153, 153)",
        'showarrow': False,
        'font': {'size': 14, 'color': "rgb(25, 25, 25)"}
    }
    lay.annotations = [annot_rel]

    :param rdf: pd.DataFrame with normalized features (use do_scaling function)
    :param plyt: plotly go.Layout object instance
    :param appd: str with domain or project name
    :param styp: str with sentiment field name to use for plot
    :return: None
    """
    twt: pd.DataFrame = rdf.copy(deep=True)
    siz_const: int = 15  # marker size: divide max value by this for sizeref
    if plyt:
        lay: go.Layout = copy.copy(plyt)
    else:
        lay: go.Layout = create_layout()

    sntdct: dict = twt[styp].value_counts().to_dict()
    sntdct = {y: sntdct[y] for y in sorted(sntdct, key=lambda x: fabs(x), reverse=True)}
    sntsum = sum(sntdct.values())

    def set_color(metric: float, segs: int = 4):
        """
        passed a float field from a dataframe, returns an rgb color to use for marker
        :param metric: float
        :return: str in form "rbg(0, 0, 0)"
        """
        sntslice: int = round(sntsum / segs, ndigits=0)
        aggr: int = 0
        for k, v in sntdct.items():
            aggr += v
            if fabs(k) < fabs(metric):
                if aggr < sntslice:
                    # return "rgb(204, 51, 51)"
                    return 1.0
                elif (aggr > sntslice):
                    # return "rgb(153, 102, 51)"
                    return 0.7
                elif (aggr > 2 * sntslice):
                    # return "rgb(102, 153, 153)"
                    return 0.3
                else:

                    # return "rgb(102, 102, 102)"
                    return 0.0
        return

    def parse_lst(coly: list):
        """
        inner fx to parse a list in a column in a dataframe
        :param coly: a list of str
        return:
        """
        if isinstance(coly, list):
            if len(coly) > 0:
                # print(" %s is list with length > 0" % coly)
                lst_str: str = ""
                for hsh in coly:
                    hsh = str(hsh).lower()
                    tmp: str = " " + hsh
                    # print(" iter item is %s" % hsh)
                    lst_str: str = lst_str + tmp
                    lst_str = lst_str.strip()
                    # print("joined creation is %s" % lst_str)
                return lst_str
            else:
                # print(" length of hash list is 0")
                return None
        else:
            # print(" did not get list from hashtag field")
            return None

    def chk_typ(coly):
        """
        if Tweet is on all three lists (Q-R-R, Fave, and Sentiment) show differently
        :param coly:
        :return:
        """
        if coly in ['qfs', 'qs', 'fs']:
            return 3
        elif coly in ['qf']:
            return 2
        elif coly in ['q', 'f']:
            return 1
        else:
            return 0

    twt['s_color'] = twt[styp].apply(lambda x: set_color(x))
    clrlst: list = twt['s_color'].to_list()
    # aspectmode options are cube, manual, data, auto
    lay.xaxis.tickmode = "linear"
    lay.xaxis.tickfont = dict(size=10)
    lay.xaxis.type="linear"
    lay.xaxis.tickformat= "%2.2f"
    lay.xaxis.showticklabels = False
    lay.title.text = "PRJX QT, RT, Reply and Like metrics<br>".replace("PRJX", appd) + \
                     "Color-coded Sentiment (light-pos, dark-neg), hover for details"
    lay.showlegend = False
    lay.scene = {'xaxis': {'title': "Tweet Wordscore", 'spikethickness': 1, 'dtick': 0.1,
                           'showtickprefix': None, 'tickformat': "%2.2f",
                           'tickfont': {'color': "rgb(51, 102, 153)",
                                                        'family': 'Helvetica Neue UltraLight',
                                                        'size': 10
                                                        }
                           },
                 'yaxis': {'title': 'Quotes-ReTweets-Replies<br>scaled', 'spikethickness': 1,
                           'showtickprefix': None,
                           'tickfont': {'color': "rgb(51, 102, 153)",
                                        'family': 'Helvetica Neue UltraLight', 'size': 14
                                        }
                           },
                 'zaxis': {'title': 'Likes - scaled', 'spikethickness': 1,
                           'tickfont': {'color': "rgb(51, 102, 153)",
                                        'family': 'Helvetica Neue UltraLight', 'size': 14
                                        }
                           },
                 'aspectmode': 'manual', 'aspectratio': {'x': 2, 'y': 1, 'z': 1},
                 }
    lay.margin.l = 100

    xlst: list = twt.wrdscore.to_list()
    qrlst: list = list(twt['qrr_log'].apply(lambda x: "{:.2f}".format(x)))
    fvlst: list = list(twt['fave_log'].apply(lambda x: "{:.2f}".format(x)))
    sntlst: list = list(twt[styp].apply(lambda x: "{: .2f}".format(x)))
    txtlst: list = list(twt['hashes'].apply(lambda x: parse_lst(x)))

    mrglst: list = []
    for qm, fm, sm in zip(qrlst, fvlst, sntlst):
        mrglst.append((float(qm) + float(fm)) * pow(float(sm), 2))
    mrg_mean = sum([float(x) for x in mrglst]) / len(mrglst)
    mrg_max = max(mrglst)
    print("shared metric has mean of %.2f and max of %.2f " % (mrg_mean, mrg_max))

    fig = go.Figure(data=go.Scatter3d(x=xlst, y=qrlst, z=fvlst,
                                      hovertemplate='<b>Tweet wordscore</b> %{x:.2f}' +
                                                    '<br>y: Scaled QRR-F: %{y:.2f}' +
                                                    "<br>Sentiment: %{customdata:.2f}" +
                                                    "<br>hashtags: %{text}",
                                      text=txtlst,
                                      name="Tweet Influence",
                                      mode='markers', showlegend=True,
                                      textsrc="%{customdata:.1f}",
                                      marker=dict(size=4,
                                                  opacity=0.8,
                                                  color=clrlst, colorscale='viridis'
                                                  ),
                                      customdata=sntlst
                                      ), layout=lay
                    )

    fig.update_layout(lay, margin=dict(l=30, r=50, b=30, t=60))
    fig.show(config=plt_cfg)

    return fig

def do_cloud(batch_tw_wrds, opt_stops: str = None, maxwrd: int=90, minlen: int=4):
    """
    wordcloud package options can be explored via '?wordcloud' (python- show docstring)
    background_color="white" - lighter background makes smaller words more legible,
    max_words= this can prevent over clutter, mask=shape the cloud to an image,
    stopwords=ad-hoc removal of unwanted words, contour_width=3,
    :param batch_tw_wrds: list of list of word tokens for tweets
    :param opt_stops: str var name for optional stop list
    :param maxwrd: integer typically 80 to 120 for maximum words to show in cloud
    :param minlen: minimum length of words allowed
    :return:
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import io

    cloud_text = io.StringIO(newline="")
    for tok in batch_tw_wrds:
        if isinstance(tok, str):
            cloud_text.write(tok + " ")
        else:
            for a_tw in tok:
                if isinstance(a_tw, list):
                    cloud_text.write(" ".join([str(x) for x in a_tw]) + " ")
                if isinstance(a_tw, str):
                    # if simple list of text for each tweet:
                    cloud_text.write(a_tw + " ")

    wordcld = WordCloud(width=800, height=800, max_words=maxwrd,
                        background_color='white',
                        stopwords=opt_stops, min_word_length=minlen, max_font_size=96,
                        min_font_size=14).generate(cloud_text.getvalue())

    # plot the WordCloud image
    # may be able to put this in plotly with the following
    # cld_img = wc.to_array()
    # go.Figure()
    # go.imshow(cld_img, interpolation="bilinear")
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcld)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    return

def plot_word_values(twl: list, plyt: go.Layout=None, appd: str=None):
    """
    plot tfidf based text scoring for tweets
    :param twl: list of dict of tweets
    :param plyt: plotly layout instance
    :param appd: a string with the project name
    :return: Plotly figure
    """
    SYMBOL_TYP: list = ["diamond", "circle", "triangle-up", "diamond", "circle"]
    lay: go.Layout = copy.copy(plyt) if plyt else create_layout()
    lay.title.text = "Workplace Dynamics - scoring of tweets using nlp-tfidf"
    lay.xaxis.titlefont = dict(size=20, color=GSC['oblk'])

    fig = go.Figure(layout=lay)
    tracex: list = [x['tdate'] for x in twl]
    tracey: list = [x['wrdscore'] for x in twl]

    fig.add_trace(go.Scatter(x=tracex, y=tracey,
                             mode='markers',
                             name="tweet smart text scoring",
                             visible=True,
                             marker=dict(symbol=SYMBOL_TYP[0], size=10,
                                         color=TRACE_COLRS[0],
                                         )),
                  )
    fig.update_layout(lay)
    fig.show(config=plt_cfg)

    return fig
