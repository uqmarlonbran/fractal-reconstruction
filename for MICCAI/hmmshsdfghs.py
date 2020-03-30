# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:55:24 2020

@author: marlo
"""

import imageio
from skimage import transform
import os
import matplotlib.patches as patches

UINT8_MAX = 2**8 - 1
UINT16_MAX = 2**16 - 1
sz = 256
nrows = 176
ncols = 256

def trim_og(a, rotate=False):
    row_diff = sz - nrows
    row_start = row_diff // 2
    row_end = sz - row_diff // 2
    a = a[30:226, :]
    if rotate: a = np.fliplr(a.T)
    return a

def trim(a, rotate=False, special=False):
    if special:
        a = a[30:226, 70:122]
    else:
        a = a[30:226, 70:121]
    if rotate: a = np.fliplr(a.T)
    return a

def trim_all(list_arrays, rotate=False, special=False):
    for i, a in enumerate(list_arrays):
        list_arrays[i] = trim(a, rotate, special)
    return list_arrays

def format_image_iters(path, sz, batch_num=0):
    if 'gancs' in path.lower():
        case = imageio.imread(path)
        case = case[batch_num*sz:(batch_num+1)*sz, :]
    elif 'can' in path.lower():
        case_pd = UINT16_MAX/UINT8_MAX * imageio.imread(path)
        return np.zeros((sz, sz)), case_pd, np.zeros((sz, sz))
    else:
        case = UINT16_MAX/UINT8_MAX * imageio.imread(path)
    case_zf = case[:, :ncols]
    case_pd = case[:, ncols:2*ncols]
    case_gt = case[:, 2*ncols:3*ncols]

    return case_zf, case_pd, case_gt

def stat_matrices():
    psnr_sample = np.zeros((len(path_stubs), len(rs)), np.float64)
    ssim_sample = np.zeros((len(path_stubs), len(rs)), np.float64)
    for i, path_stub in enumerate(path_stubs):
        if 'gancs' in path_stub.lower():
            path_ex = ex_gancs
            case_num = case_num_gancs
            stat_name = stat_name_gancs.format(case_num, case_num)
        else:
            path_ex = ex_iters
            case_num = case_num_iters
            stat_name = stat_name_iters.format(case_num)

        for j, r in enumerate(rs):
            if 'gancs' in path_stub.lower():
                path_ex_ = path_ex.format(r, best_folds[j])
            else:
                path_ex_ = path_ex.format(r)
            path = os.path.join(path_stub, path_ex_, stat_name)
            eval_df = pd.read_csv(path, index_col=None, header=0, sep=' ')
            psnr_sample[i, j] = eval_df.at[case_num_iters, 'PSNR']
            ssim_sample[i, j] = eval_df.at[case_num_iters, 'SSIM']
        
    return psnr_sample, ssim_sample

###
path_zf = 'zf/'
path_nlcg = 'nlcg/'
path_mlem = 'mlem/'
path_sirt = 'sirt/'
path_gancs = 'GANCS/'


ex_iters = 'test_{}_best/'
image_name_iters = '{}_compare.png'
stat_name_iters = 'test_stats.csv'
case_num_iters = 157  # 12 (6-0), 33 (16-1), 157 (78-1)

ex_gancs = 'test_{}_{}/'
image_name_gancs = 'batch0000{}_test00{}.png'
stat_name_gancs = '79_test_stats.csv'
case_num_gancs = case_num_iters // 2
batch_num = 1

path_stubs = [path_zf, path_nlcg, path_mlem, path_gancs, path_can]
rs = [2, 4, 8]
best_folds = [1, 1, 1]
###

###
psnr_sample, ssim_sample = stat_matrices()
print(psnr_sample)
print(ssim_sample)
###

###
mega_con = np.ndarray(0)
first = True
for i, path_stub in enumerate(path_stubs):
    if 'gancs' in path_stub.lower():
        path_ex = ex_gancs
        case_num = case_num_gancs
        image_name = image_name_gancs.format(case_num, case_num)
    else:
        path_ex = ex_iters
        case_num = case_num_iters
        image_name = image_name_iters.format(case_num)
    r_gt = np.ndarray(0)
    r_conc = np.ndarray(0)
    r_first = True
    for j, r in enumerate(rs):
        if 'gancs' in path_stub.lower():
            path_ex_ = path_ex.format(r, best_folds[j])
        else:
            path_ex_ = path_ex.format(r)
        path = os.path.join(path_stub, path_ex_, image_name)
        zf, pr, gt = format_image_iters(path, sz, batch_num)
        if i == len(path_stubs) - 1:
            [zf, pr] = trim_all([zf, pr], rotate=True, special=True)
        else:
            [zf, pr] = trim_all([zf, pr], rotate=True)
        gt = trim_og(gt, rotate=False)
        if r_first:
            r_gt = gt
            r_conc = pr
            r_first = False
        else:
            r_conc = np.concatenate([r_conc, pr], axis=1)
    conc = r_conc  # np.concatenate([r_gt, r_conc], axis=1)
    if first:
        big_gt = r_gt
        mega_conc = conc
        first = False
    else:
        mega_conc = np.concatenate([mega_conc, conc], axis=0)

# bgcols = mega_conc.shape[0]
# bgrows = int(nrows*bgcols/ncols)
# big_gt = transform.resize(r_gt, (bgrows, bgcols), mode='symmetric',
#                           preserve_range=True)
# big_gt = transform.rotate(big_gt, 90, resize=True)
big_gt = np.fliplr(big_gt.T)
        
print(mega_conc.shape)
print(big_gt.shape)
mega_conc = np.concatenate([big_gt, mega_conc], axis=1)
print(mega_conc.shape)
###

###
mega_conc[102:153, 196:] = 0
###

###
plt.gray()
fig = plt.figure(frameon=False)
template_width_in = 12.2/2.54 - 0.05  # Springer LNCS textwidth converted to cm, with padding
h = template_width_in * mega_conc.shape[0] / mega_conc.shape[1]
dpi = mega_conc.shape[1] / template_width_in
fig.set_size_inches(template_width_in, h)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# Text overlay
font1 = {'family': 'serif',
        'color':  'cyan',
        'weight': 'normal',
        'horizontalalignment': 'left',
        'verticalalignment': 'top',
        'size': 6,
        }
font2 = {'family': 'serif',
        'color':  'cyan',
        'weight': 'normal',
        'horizontalalignment': 'right',
        'verticalalignment': 'top',
        'size': 6,
        }
x01 = 200
y01 = 0
dy1 = 51
ax.text(x01, y01+3, 'ZF', fontdict=font1)
ax.text(x01, y01+dy1, 'CS-WV', fontdict=font1)
ax.text(x01, y01+2*dy1, 'fMLEM', fontdict=font1)
ax.text(x01, y01+3*dy1, 'fSIRT', fontdict=font1)
ax.text(x01, y01+4*dy1, 'GANCS', fontdict=font1)

x02 = 390
dx2 = 196
y02 = 0
dy2 = 51
x2 = x02
for i in range(3):
    for j in range(5):
        if j == 2: continue
        text = f'{psnr_sample[j,i]:.1f}' + f'\n{ssim_sample[j,i]:.3f}'.replace('0.', '.')
        if j == 0:
            ax.text(x2, y02+j*dy2+3, text, fontdict=font2)
        else:
            ax.text(x2, y02+j*dy2, text, fontdict=font2)
    x2 += dx2

# Add crop outline
left, width = 0, .25
bottom, height = .5273, .1992
right = left + width
top = bottom + height
p = patches.Rectangle(
    (left, bottom), width, height, 
    fill=False, transform=ax.transAxes, clip_on=False,
    edgecolor='cyan', linestyle=':', linewidth=.75,
    )
ax.add_patch(p)

ax.imshow(mega_conc, aspect='auto')
plt.savefig(f'./visual/cart1d/fractal_cs_gallery_{case_num_iters}.eps', format='eps', dpi=dpi)
plt.savefig(f'./visual/cart1d/fractal_cs_gallery_{case_num_iters}.png', format='png', dpi=dpi)
###