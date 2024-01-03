import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
import matplotlib.cm as cm
import math
from matplotlib.patches import Arc
warnings.filterwarnings('ignore')



def preprocess():
    '''
    预处理数据, 添加Continent列
    '''
    
    country_code_fil = "/mnt/c/Users/Administrator.DESKTOP-HE1VVOU/Desktop/Radial_histogram/data/country_code.csv"
    country_code = pd.read_csv(country_code_fil, sep=',')
    
    crop_price_fil = "/mnt/c/Users/Administrator.DESKTOP-HE1VVOU/Desktop/Radial_histogram/data/crop_price_movmean_fillna_mai.csv"
    crop_price = pd.read_csv(crop_price_fil, sep=',')
    crop_price.insert(3, 'Continent', None)
    
    for code in crop_price['AreaCode_M49_']:
        if code not in country_code['M49_Code'].values:
            print (code, crop_price.loc[crop_price['AreaCode_M49_'] == code, 'Area'].values[0])
            continue
        crop_price.loc[crop_price['AreaCode_M49_'] == code, 'Continent'] = country_code.loc[country_code['M49_Code'] == code, 'Continent_EN_Name'].values[0]
    
    crop_price.loc[crop_price['AreaCode_M49_'] == 0, 'Continent'] = 'Global'
    crop_price.to_csv("/mnt/c/Users/Administrator.DESKTOP-HE1VVOU/Desktop/Radial_histogram/data/crop_price_movmean_fillna_mai_continent.csv", index=False)
    
    return crop_price




def Radial_histogram(data_in, pry_cat_colname, sec_cat_colname, data_levels, **kwargs):
    '''
    
                --- 此函数用于绘制极坐标堆叠条形图 --- 
    数据分三级，一级为主要分类，二级为次要分类，三级为数据级别(堆叠分类)。
    
    必选参数：
        data_in             (DataFrame) : 包含数据的DataFrame                            
        pry_cat_colname           (str) : 主要分类的列名                             
        sec_cat_colname           (str) : 次要分类的列名                             
        data_levels              (list) : 数据级别的列名列表
        
    可选参数：                            
        primary_cats             (list) : 主要分类的列表。默认为数据中的所有唯一主要分类              
        secondary_cats           (list) : 次要分类的列表。默认为数据中的所有唯一次要分类              
        inner_circle_radius     (float) : 内圆的半径。默认为0                            
        blank_length              (int) : 每个主要分类之间的空白条形数。默认为2                 
        levels_color             (list) : 每个数据级别的颜色。默认为蓝色调色板的颜色                
        radii                    (list) : 每个数据级别的半径。默认为数据级别的最大值和总和的最大值之间的5个等距值 
        ylims                    (list) : [ymin, ymax], y轴的最小值和最大值。默认为数据级别的最大值和总和的最大值
        sort_by_Total            (bool) : 是否按总和对次要分类进行排序。默认为True               
        sort_ascending           (bool) : 是否按升序对次要分类进行排序。默认为False              
        bar_linestyle             (str) : 条形的线条样式。默认为虚线                       
        bar_linewidth           (float) : 条形的线条宽度。默认为1                          
        bar_edgecolor             (str) : 条形的边缘颜色。默认为白色                       
        bar_alpha               (float) : 条形的透明度。默认为1                           
        circle_label_fontsize     (int) : 圆圈标签的字体大小。默认为10                     
        circle_label_fontcolor    (str) : 圆圈标签的字体颜色。默认为黑色   
        circle_label_fontweight   (str) : 圆圈标签的字体粗细。默认为normal                  
        circle_linestyle          (str) : 圆圈的线条样式。默认为虚线                       
        circle_linewidth        (float) : 圆圈的线条宽度。默认为1                          
        circle_edgecolor          (str) : 圆圈的边缘颜色。默认为灰色                       
        circle_alpha            (float) : 圆圈的透明度。默认为1                           
        circle_fill              (bool) : 是否填充圆圈。默认为False                      
        bottom_circle_linestyle   (str) : 底部圆圈的线条样式。默认为实线                     
        bottom_circle_linewidth (float) : 底部圆圈的线条宽度。默认为2                        
        bottom_circle_linecolor   (str) : 底部圆圈的线条颜色。默认为黑色                     
        pry_fontsize              (int) : 主要分类标签的字体大小。默认为13                   
        pry_fontcolor             (str) : 主要分类标签的字体颜色。默认为黑色       
        pry_fontweight            (str) : 主要分类标签的字体粗细。默认为bold            
        sec_fontsize              (int) : 次要分类标签的字体大小。默认为10                   
        sec_fontcolor             (str) : 次要分类标签的字体颜色。默认为黑色
        pry_fontweight            (str) : 次要分类标签的字体粗细。默认为normal                   
        title                     (str) : 图表的标题                               
        title_fontsize            (int) : 图表标题的字体大小。默认为15                     
        title_fontcolor           (str) : 图表标题的字体颜色。默认为黑色 
        title_fontweight          (str) : 图表标题的字体粗细。默认为normal
        legend_on                (bool) : 是否显示图例。默认为True     
        legend_label_fontsize    (int) : 图例标签的字体大小。默认为10               
        legend_bbox              (list) : 图例的位置[横坐标，纵坐标]。默认为[0.5, 0.5]。[0,0]为左下角，[1,1]为右上角
        offset_pry_text         (float) : 主要分类标签的偏移量。默认为-5
        offset_inner            (float) : 内圆圈的偏移量。默认为-2
    '''

    primary_cats = kwargs.get('primary_cats', data_in[pry_cat_colname].unique())
    secondary_cats = kwargs.get('secondary_cats', data_in[sec_cat_colname].unique())
    radii = kwargs.get('radii', None)
    ylims = kwargs.get('ylims', None)
    title = kwargs.get('title', None)
    levels_color = kwargs.get('levels_color', None)
    inner_circle_radius = kwargs.get('inner_circle_radius', 0)
    blank_length = kwargs.get('blank_length', 2)
    sort_ascending = kwargs.get('sort_ascending', False)
    sort_by_Total = kwargs.get('sort_by_Total', True)
    bar_linestyle = kwargs.get('bar_linestyle', '--')
    bar_linewidth = kwargs.get('bar_linewidth', 1)
    bar_edgecolor = kwargs.get('bar_edgecolor', 'white')
    bar_alpha = kwargs.get('bar_alpha', 1)
    circle_linestyle = kwargs.get('circle_linestyle', '--')
    circle_linewidth = kwargs.get('circle_linewidth', 1)
    circle_edgecolor = kwargs.get('circle_edgecolor', 'grey')
    circle_alpha = kwargs.get('circle_alpha', 1)
    circle_fill = kwargs.get('circle_fill', False)
    bottom_circle_linestyle = kwargs.get('bottom_circle_linestyle', '-')
    bottom_circle_linewidth = kwargs.get('bottom_circle_linewidth', 2)
    bottom_circle_linecolor = kwargs.get('bottom_circle_linecolor', 'black')
    pry_fontsize = kwargs.get('pry_fontsize', 13)
    sec_fontsize = kwargs.get('sec_fontsize', 10)
    title_fontsize = kwargs.get('title_fontsize', 15)
    circle_label_fontsize = kwargs.get('circle_label_fontsize', 10)
    pry_fontcolor = kwargs.get('pry_fontcolor', 'black')
    sec_fontcolor = kwargs.get('sec_fontcolor', 'black')
    title_fontcolor = kwargs.get('title_fontcolor', 'black')
    bar_label_fontcolor = kwargs.get('bar_label_fontcolor', 'black')
    circle_label_fontcolor = kwargs.get('circle_label_fontcolor', 'black')
    pry_fontweight = kwargs.get('pry_fontweight', 'bold')
    sec_fontweight = kwargs.get('sec_fontweight', 'normal')
    title_fontweight = kwargs.get('title_fontweight', 'normal')
    circle_label_fontweight = kwargs.get('circle_label_fontweight', 'normal')
    legend_on = kwargs.get('legend_on', True)
    legend_label_fontsize = kwargs.get('legend_label_fontsize', 10)
    legend_bbox = kwargs.get('legend_bbox', [0.5, 0.5])
    offset_pry_text = kwargs.get('offset_pry_text', -5)
    offset_inner = kwargs.get('offset_inner', -2)
    
    
    # 计算主要和次要分类的数量
    n_pry = len(primary_cats)
    n_sec = len(secondary_cats)

    # 计算数据级别的最大值和总和的最大值
    max_level = data_in[data_levels].max().max()
    data_in['total'] = data_in[data_levels].sum(axis=1)
    max_sum = data_in['total'].max()
    min_sum = data_in['total'].min()
    
    # 向上取整到最接近的 10 的倍数
    if ylims is None:
        ymax = math.ceil(max_sum / 10) * 10
        ymin = math.floor(min_sum / 10) * 10
    else:
        ymin, ymax = ylims
        
    if radii is None:
        radii = np.linspace(ymin, ymax, 5).tolist()
    
    # 计算每个条形的宽度
    width_per_bar = (2 * np.pi) / (n_sec + ((n_pry+1) * blank_length))
    
    # 创建极坐标图
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) 

    # 创建颜色级别
    if levels_color is None:
        levels_color = sns.color_palette("Blues_r", len(data_levels))
    else:
        if len(levels_color)<len(data_levels):
            raise ValueError('levels_color must have at least as many colors as data_levels')
            levels_color = sns.color_palette("Blues_r", len(data_levels))

    # 绘制每个数据级别的圆圈
    for radius in radii:
        circle = plt.Circle((0, 0), radius + inner_circle_radius, transform=ax.transData._b, color=circle_edgecolor, 
                            fill=circle_fill, linestyle=circle_linestyle, linewidth=circle_linewidth, alpha=circle_alpha)
        ax.add_artist(circle)
    
    # 在每个圆圈旁添加文本标签
    for radius in radii:
        ax.text(0, radius + inner_circle_radius, str(radius), ha='center', va='center', color=circle_label_fontcolor,
                fontsize=circle_label_fontsize, rotation_mode='anchor', fontweight=circle_label_fontweight)
            
    # 初始化起始角度
    angle = width_per_bar * (blank_length+1)

    # 绘制每个主要和次要分类的条形图
    for primary_cat in primary_cats:
        if sort_by_Total:
            primary_cat_data = data_in[data_in[pry_cat_colname] == primary_cat].sort_values(by=['total'], ascending=sort_ascending)
        else:
            primary_cat_data = data_in[data_in[pry_cat_colname] == primary_cat]
        sec_agl = []
        for secondary_cat in primary_cat_data[sec_cat_colname].unique():
            secondary_cat_data = primary_cat_data[primary_cat_data[sec_cat_colname] == secondary_cat]
            bottom = inner_circle_radius
            for j, data_level in enumerate(data_levels):
                value = secondary_cat_data[data_level].sum()
                ax.bar(angle, value, width=width_per_bar, color=levels_color[j], bottom=bottom, 
                       edgecolor=bar_edgecolor, linewidth=bar_linewidth, alpha=bar_alpha, linestyle=bar_linestyle, label=data_level)
                bottom += value
            # 添加与bar平行的text
            text_angle_deg = -np.degrees(angle)+90
            alignment = {'va': 'center', 'ha': 'left'}
            # 检查文本是否位于圆的下半部分
            if text_angle_deg < -90 and text_angle_deg >= -270:
                text_angle_deg += 180
                alignment['ha'] = 'right'
            # 添加次要分类标签
            ax.text(angle, inner_circle_radius, secondary_cat, rotation=text_angle_deg, rotation_mode='anchor', **alignment, 
                    fontsize=sec_fontsize, fontweight=sec_fontweight, color=sec_fontcolor)            
            
            sec_agl.append(angle)
            angle += width_per_bar
        angle += width_per_bar * blank_length
        # 在每个主要分类旁添加文本标签
        if len(sec_agl):
            angles = np.linspace(sec_agl[0]-width_per_bar/2, sec_agl[-1]+width_per_bar/2, 100)
            ax.plot(angles, [inner_circle_radius+offset_inner] * len(angles), color=bottom_circle_linecolor, linewidth=bottom_circle_linewidth)
            center_angle = np.mean(sec_agl)
            text_angle_deg = -np.degrees(center_angle)
            alignment = {'va': 'center', 'ha': 'center'}
            # 检查文本是否位于圆的下半部分
            if text_angle_deg < -90 and text_angle_deg >= -270:
                text_angle_deg += 180
            # 添加主要分类标签
            ax.text(center_angle, inner_circle_radius+offset_pry_text, primary_cat, 
                        rotation=text_angle_deg, **alignment,
                        rotation_mode='anchor', fontsize=pry_fontsize,fontweight=pry_fontweight, color=pry_fontcolor)

    ax.set_ylim(ymin, ymax + inner_circle_radius)
    if title is not None:
        plt.title(title, fontsize=title_fontsize, color=title_fontcolor, fontweight=title_fontweight)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in levels_color]
    labels = data_levels
    if legend_on:
        plt.legend(handles, labels, loc='center', bbox_to_anchor=legend_bbox, fontsize=legend_label_fontsize)
    fig.tight_layout(pad=3.0)
    plt.grid(False)
    plt.axis('off')
    plt.show()




def main():
    crop_price = preprocess()
    Continents = crop_price['Continent'].unique()
    print(Continents)
    Continents = [ 'Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    Continents = [ 'Africa', 'Oceania', 'Asia', 'North America', 'Europe', 'South America']
    
    years = ['Y1991', 'Y1992', 'Y1993', 'Y1994']
    
     # 对所有数据级别取对数
    for year in years:
        crop_price[year] = np.log(crop_price[year] + 1)  # 添加1以避免log(0)
    
    # --------  crop_price 必须经过预处理，添加了Continent列
    Radial_histogram(crop_price, 'Continent', 'Area', years, ylims=[0, 30], radii=[0, 5, 10, 15, 20], inner_circle_radius=30, 
                     blank_length=3, primary_cats=Continents, pry_fontsize=15, sec_fontsize=12)
    




if __name__ == "__main__":
    main()