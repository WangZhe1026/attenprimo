import matplotlib.pyplot as plt

data = {
    "columns": ["dim", "attn"],
    "data": [
        [10.0, 1.5892808846729167e-08], [20.0, 2.6787392926053144e-07], [30.0, 5.262335889710812e-06],
        [40.0, 1.530197778265574e-06], [50.0, 0.00019854064157698303], [60.0, 0.00011245330097153783],
        [70.0, 0.001105316448956728], [80.0, 0.0015679742209613323], [90.0, 0.0005584206082858145],
        [100.0, 0.002997433999553323], [110.0, 0.003654778702184558], [120.0, 0.002217173809185624],
        [130.0, 0.004018761683255434], [140.0, 0.04878443479537964], [150.0, 0.03103148192167282],
        [160.0, 0.0742490291595459], [170.0, 0.10168251395225525], [180.0, 0.15661509335041046],
        [190.0, 0.24472324550151825], [200.0, 0.3264763355255127]
    ]
}

# 提取数据
dims = [row[0] for row in data["data"]]
attn_values = [row[1] for row in data["data"]]

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(dims, attn_values, marker='o', color='b', linestyle='-')

# 添加标题和标签
plt.title('Attention Values vs. Dimension')
plt.xlabel('Dimension')
plt.ylabel('Attention Value')

# 显示图例
plt.legend(['Attention Values'])

# 显示网格线
plt.grid(True)

# 显示图形
plt.show()