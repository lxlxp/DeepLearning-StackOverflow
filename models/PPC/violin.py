import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#设置绘图风格
plt.style.use('ggplot')
#处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#坐标轴负号的处理
plt.rcParams['axes.unicode_minus']=False

# 读取数据
tips = pd.read_excel(r'phase0.xlsx')
#print(tips)
# 绘制分组小提琴图
plt.figure (figsize= (6,6))
sns.violinplot(
               x = "phase", # 指定x轴的数据
               y = "probability", # 指定y轴的数据
               data = tips, # 指定绘图的数据集
            #    order = ['0','1','2','3','4','5'], # 指定x轴刻度标签的顺序
               scale = "width", # 调节宽度
               palette = 'RdBu' ,# 指定颜色
              )
plt.yticks( size=15,weight='bold')#设置大小及加粗
plt.xticks( size=15,weight='bold')
plt.xlabel('phase',fontsize = 15,fontweight='bold')
plt.ylabel('probability',fontsize = 15,fontweight='bold')
# 添加图形标题
plt.title('Basic Knowledge',fontsize = 20,fontweight='bold')
# 设置图例
plt.legend(loc = 'upper right', ncol = 2)
#控制横纵坐标的值域
plt.axis([-1,6,0,1])
# 显示图形
plt.savefig("BK.pdf")