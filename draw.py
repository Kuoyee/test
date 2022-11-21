
from pyecharts.charts.basic_charts import bar
from pyecharts.charts.basic_charts import line
from pyecharts.charts.basic_charts import pie
from pyecharts.charts.basic_charts import radar
from pyecharts import options as opts
from pyecharts.globals import ThemeType
def draw_logistic_radar(list):
    value=list
    c_schema = [
        {"name": "Area", "max": 1, "min": -1},
        {"name": "District_id", "max": 1, "min": -1},
        {"name": "Decoration", "max": 1, "min": -1},
        {"name": "Bathroom", "max": 1, "min": -1},
        {"name": "Hall", "max": 1, "min": -1},
        {"name": "Floor", "max": 1, "min": -1},
        {"name": "TV", "max": 1, "min": -1},
        {"name": "Orientation", "max": 1, "min": -1},
        {"name": "Airconditioner", "max": 1, "min": -1},
        {"name": "Waterheater", "max": 1, "min": -1},
        {"name": "Room", "max": 1, "min": -1}]
    mradar=(
        radar.Radar(init_opts=opts.InitOpts(bg_color='#FFFFFF', theme=ThemeType.INFOGRAPHIC))
            .add_schema(schema=c_schema, shape="circle",
                        splitarea_opt=opts.SplitAreaOpts(
            is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)))
            .add("标准化系数", value[1], linestyle_opts=opts.LineStyleOpts(width=3))
            .add("标准误差", value[0], linestyle_opts=opts.LineStyleOpts(width=3),color="#4169E1")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title="回归分析变量系数情况"),legend_opts=opts.LegendOpts())
            .render("logisticresult.html")
    )
def draw_decoration_bar(list):
    attr = ["简单装修","精装修"]
    mbar = (
        bar.Bar(init_opts=opts.InitOpts(bg_color='#FFFFFF',theme=ThemeType.INFOGRAPHIC))  # 生成line类型图表
            .add_xaxis(attr)  # 添加x轴，Faker.choose()是使用faker的随机数据生成x轴标签
            .add_yaxis('平均值', list)
            .set_global_opts(title_opts=opts.TitleOpts(title='各装修级别单位平面平均价格'))
    )
    mbar.render('decoration.html')
def draw_decorationcount_pie(lista):
    attr = ["海沧", "湖里", "集美", "思明", "翔安", "同安"]
    data_pair = [list(z) for z in zip(attr,lista)]
    #"厦门市各区域精装修出租房屋数量饼状图", title_pos='center',
    mpie = (
        pie.Pie( init_opts=opts.InitOpts(bg_color='#FFFFFF', theme=ThemeType.INFOGRAPHIC))
            .add(series_name="精装修出租房屋数量",data_pair=data_pair,)
            .set_global_opts(title_opts=opts.TitleOpts(title="厦门市各区域精装修\n出租房屋数量饼状图"))
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}:{d}%")))
    mpie.render(path="decorationcount.html")

def draw_orientationcount_pie(lista):
    attr = ["海沧", "湖里", "集美", "思明", "翔安", "同安"]
    data_pair = [list(z) for z in zip(attr,lista)]
    #"厦门市各区域朝南出租房屋数量饼状图", title_pos='center',
    mpie = (
        pie.Pie( init_opts=opts.InitOpts(bg_color='#FFFFFF', theme=ThemeType.INFOGRAPHIC))
            .add(series_name="精装修出租房屋数量",data_pair=data_pair,)
            .set_global_opts(title_opts=opts.TitleOpts(title="厦门市各区域朝南\n出租房屋数量饼状图"))
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}:{d}%")))
    mpie.render(path="orientationcount.html")
def draw_area_bar(all_list):
    all_list[0] = ['{:.2f}'.format(i) for i in all_list[0]]
    all_list[0] = [float(i) for i in all_list[0]]
    all_list[1] = ['{:.2f}'.format(i) for i in all_list[1]]
    all_list[1] = [float(i) for i in all_list[1]]
    all_list[2] = ['{:.2f}'.format(i) for i in all_list[2]]
    all_list[2] = [float(i) for i in all_list[2]]
    all_list[3] = ['{:.2f}'.format(i) for i in all_list[3]]
    all_list[3] = [float(i) for i in all_list[3]]
    attr = ["海沧", "湖里", "集美", "思明", "翔安", "同安"]
    v0 = all_list[0]
    v1 = all_list[1]
    v2 = all_list[2]
    v3 = all_list[3]
    # "厦门市租房面积概况",
    mbar = (
        bar.Bar(init_opts=opts.InitOpts(bg_color='#FFFFFF', theme=ThemeType.INFOGRAPHIC))  # 生成line类型图表
            .add_xaxis(attr)  # 添加x轴，Faker.choose()是使用faker的随机数据生成x轴标签
            .add_yaxis('最小值', v0, label_opts=opts.LabelOpts(is_show=True))  # 添加y轴
            .add_yaxis('最大值', v1, label_opts=opts.LabelOpts(is_show=True))
            .add_yaxis('平均值', v2, label_opts=opts.LabelOpts(is_show=True))
            .add_yaxis('中位数', v3, label_opts=opts.LabelOpts(is_show=True))
            .set_global_opts(title_opts=opts.TitleOpts(title='厦门市各区租房面积概况'))
    )
    mbar.render('area.html')

def draw_District_line(all_list):
    all_list[0] = ['{:.2f}'.format(i) for i in all_list[0]]
    all_list[0] = [float(i) for i in all_list[0]]
    all_list[1] = ['{:.2f}'.format(i) for i in all_list[1]]
    all_list[1] = [float(i) for i in all_list[1]]
    all_list[2] = ['{:.2f}'.format(i) for i in all_list[2]]
    all_list[2] = [float(i) for i in all_list[2]]
    all_list[3] = ['{:.2f}'.format(i) for i in all_list[3]]
    all_list[3] = [float(i) for i in all_list[3]]
    print("开始绘图")
    attr = ["海沧", "湖里", "集美", "思明", "翔安", "同安"]
    v0 = all_list[0]
    v1 = all_list[1]
    v2 = all_list[2]
    v3 = all_list[3]
#"厦门市租房租金概况",
    mline = (
        line.Line( init_opts=opts.InitOpts(bg_color='#FFFFFF',theme=ThemeType.INFOGRAPHIC))  # 生成line类型图表
            .add_xaxis(attr)  # 添加x轴，Faker.choose()是使用faker的随机数据生成x轴标签
            .add_yaxis('最小值', v0, label_opts=opts.LabelOpts(is_show=True), linestyle_opts=opts.LineStyleOpts(width=3))  # 添加y轴
            .add_yaxis('最大值', v1, label_opts=opts.LabelOpts(is_show=True),linestyle_opts=opts.LineStyleOpts(width=3))
            .add_yaxis('平均值', v2, label_opts=opts.LabelOpts(is_show=True),linestyle_opts=opts.LineStyleOpts(width=3))
            .add_yaxis('中位数', v3, label_opts=opts.LabelOpts(is_show=True),linestyle_opts=opts.LineStyleOpts(width=3))
            .set_global_opts(title_opts=opts.TitleOpts(title='厦门市租房租金概况'))
    )
    mline.render('district.html')
    print("结束绘图")

value=[
    [[0.003,0.086,0.276,0.205,0.224,0.010,0.326,0.084,0.491,0.473,0.154]],
    [[0.689,0.337,0.119,0.111,0.085,0.035,0.025,-0.023,0.062,-0.047,0.031]]
]
draw_logistic_radar(value)