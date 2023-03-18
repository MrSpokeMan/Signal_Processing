import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from PIL import Image

res = 256 / 96 * 2  # resolution
M = 64  # frames in one circle of animation
L = 32  # amount of layers divided by 2
n = 5  # quantity of propellers
j = 0


# function making aliasing effect
def aliasing():
    global j
    fig1.savefig('tmpIMG.png')
    tmpIMG = Image.open('tmpIMG.png')
    tmpCrop = tmpIMG.crop(box=(0, j * L, 512, 512))
    background.paste(tmpCrop, (0, j * L))
    j += 1
    if round(j > 512 / L):
        background.save(f'{n}-prop-{L / 2}-lines.png')
        background.show()
        exit()


fig1, ax = plt.subplots(
    subplot_kw={'projection': 'polar'}, figsize=(res, res))
background = Image.new('RGB', size=(512, 512), color='white')

# saving the arguments "X" in an array
rads = np.arange(0, 2 * np.pi, .001)
# setting "Y" coordinates
r = np.sin(n * rads + np.pi)

line, = ax.plot(rads, r)

ax.set_rticks([])
ax.set_rlim(bottom=-1, top=1)


def animate(i):
    if i > 1:
        aliasing()
    line.set_ydata(np.sin(n * rads + i * np.pi / 20))
    return line,


aniPolar = ani.FuncAnimation(fig1, animate, interval=M, repeat=False)

plt.show()
