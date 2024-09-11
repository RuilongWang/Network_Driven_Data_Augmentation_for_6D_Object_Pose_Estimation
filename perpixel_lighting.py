import numpy as np
import skimage.io
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

n = np.load('normal_map.npy')
d = np.load('depth_map.npy')

n = n / 255.0

material_color = skimage.io.imread(r'C:\Users\Ruilo\PycharmProjects\DecepetionNetConda\dataset\lm_train\train\000002\rgb\000827.png')
material_color = material_color / 255.0

n = n.reshape(n.shape[0]*n.shape[1], n.shape[2])
print(n.shape)
n = normalize(n, axis=1).ravel()
n = n.reshape(480, 640, 3)

x = np.arange(n.shape[0])
print(x.shape)
y = np.arange(n.shape[1])


fx=572.4114
fy=573.57043
cx=325.2611
cy=242.04899


# for i in range(d.shape[0]):
#     for j in range(d.shape[1]):
#         P3D[i,j,0] = (i - cx) * d[i,j] / fx
#         P3D[i,j,1] = (j - cy) * d[i,j] / fy
#         P3D[i,j,2] = d[i,j]


P3D_x = np.repeat((x-cx)[:,np.newaxis], n.shape[1],axis = 1) * d / fx
P3D_y = np.repeat((y-cy)[np.newaxis,:], n.shape[0],axis = 0) * d / fy
P3D_z = d
P3D = np.concatenate((P3D_x[:,:,np.newaxis],P3D_y[:,:,np.newaxis],P3D_z[:,:,np.newaxis]),axis = 2)

e_pos = np.array([0,0,400])
e = e_pos - P3D

e = normalize(e.reshape(480*640,3), axis=1).ravel()
e = e.reshape(480,640,3)


light_dir = np.array([0,1,0])
light_dir = normalize(light_dir[:,np.newaxis], axis=0).ravel()
print('light', light_dir.shape)

light_color = np.array([1,1,1])
shininess = 10

# print(material_color)

ambient_color=diffuse_color=specular_color = light_color * material_color
# a = np.dot(n,light_dir)
# print(np.amax(n))

intensity = np.maximum(np.dot(n,light_dir), 0.0)
intensity = np.repeat(intensity[:, :, np.newaxis], 3, axis=2)
print('intensity',intensity.shape)
out_color = np.zeros_like(material_color)
spec = np.zeros_like(material_color)

half = light_dir+e
half = normalize(half.reshape(480*640,3), axis=1).ravel()
half = half.reshape(480,640,3)

print(e[240,320,:])
print(half[240,320,:])
print(n[240,320,:])

intSpec = np.maximum(np.sum(n * half,axis=2), 0.0)
print(intSpec[240,320])
spec = specular_color * np.repeat(np.power(intSpec, shininess)[:, :, np.newaxis], 3, axis=2)
# spec = specular_color * np.power(intSpec, shininess)
out_color = 1 * ambient_color + 1 * intensity * diffuse_color + 1 * spec


plt.imshow(out_color)
plt.show()
