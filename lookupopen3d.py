import open3d as o3d
import numpy as np
# 加载 OBJ 文件
mesh1 = o3d.io.read_triangle_mesh("dog_01.obj")
mesh2 = o3d.io.read_triangle_mesh("dog_02.obj")

# 设置网格的边缘线颜色为黑色
mesh1.compute_vertex_normals()
mesh2.compute_vertex_normals()
mesh1.paint_uniform_color([0.8, 0.8, 0.8])
mesh2.paint_uniform_color([0.8, 0.8, 0.8])



# 对一个 mesh 进行平移
translation_vector = np.array([1, 0, 0])
mesh2.translate(translation_vector)

# 创建一个可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加网格到可视化窗口
vis.add_geometry(mesh1)
vis.add_geometry(mesh2)

# 设置网格显示为线框模式
opt = vis.get_render_option()
opt.mesh_show_wireframe = True

# 设置视角
vis.get_view_control().set_front([0, 0, -1])
vis.get_view_control().set_up([0, 1, 0])



line_set = o3d.geometry.LineSet()
points = np.vstack([mesh1.vertices[2424], mesh2.vertices[2362], mesh1.vertices[38], mesh2.vertices[39]])

lines = [[0, 1], [2, 3]]
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)

# 将颜色向量转换为 Vector3dVector 类型
colors = [[0, 0, 0] for i in range(len(lines))]
line_set.colors = o3d.utility.Vector3dVector(colors)

vis.add_geometry(line_set)

# 渲染并展示
vis.run()
vis.destroy_window()