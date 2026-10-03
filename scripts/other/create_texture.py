import xarray as xr
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix
from skimage.feature import graycoprops


# ==========================================================
# CONFIG
# ==========================================================

SAR_NC = "SAR_1.nc"

OUTPUT = "texture_1.parquet"

WINDOW = 11
HALF = WINDOW // 2

STRIDE = 40
LEVELS = 32


VH_MIN = -60
VH_MAX = 0


VV_MIN = -60
VV_MAX = 0


ANGLES = [
    0,
    np.pi/4,
    np.pi/2,
    3*np.pi/4
]


# ==========================================================
# LOAD NC
# ==========================================================

print("Loading nc...")

ds = xr.open_dataset(SAR_NC)

vh = ds["Sigma0_VH_db"].values
vv = ds["Sigma0_VV_db"].values

lat = ds["lat"].values
lon = ds["lon"].values


nrows, ncols = vh.shape


# ==========================================================
# HELPERS
# ==========================================================

def quantize(img, vmin, vmax):

    q = (img - vmin) / (vmax - vmin)

    q = np.clip(q, 0, 1)

    q *= (LEVELS - 1)

    return q.astype(np.uint8)


def compute_glcm_features(patch, vmin, vmax):

    q = quantize(
        patch,
        vmin,
        vmax
    )

    glcm = graycomatrix(

        q,

        distances=[1],

        angles=ANGLES,

        levels=LEVELS,

        symmetric=True,

        normed=True

    )



    contrast = graycoprops(
        glcm,
        'contrast'
    ).mean()



    homogeneity = graycoprops(
        glcm,
        'homogeneity'
    ).mean()



    correlation = graycoprops(
        glcm,
        'correlation'
    ).mean()



    asm = graycoprops(
        glcm,
        'ASM'
    ).mean()



    ent = []


    for i in range(len(ANGLES)):

        P = glcm[:, :, 0, i]


        e = -np.sum(

            P * np.log2(

                P + 1e-12

            )

        )


        ent.append(e)



    entropy = np.mean(ent)


    return (

        contrast,

        homogeneity,

        asm,

        correlation,

        entropy

    )



# ==========================================================
# PROCESS
# ==========================================================

results = []

count = 0


for r in range(

        HALF,

        nrows - HALF,

        STRIDE

):


    if r % 1000 == 0:
        print("Row", r)



    for c in range(

            HALF,

            ncols - HALF,

            STRIDE

    ):



        vh_patch = vh[

            r - HALF:r + HALF + 1,

            c - HALF:c + HALF + 1

        ]


        vv_patch = vv[

            r - HALF:r + HALF + 1,

            c - HALF:c + HALF + 1

        ]



        if not np.isfinite(vh_patch).all():
            continue


        if not np.isfinite(vv_patch).all():
            continue




        vh_features = compute_glcm_features(

            vh_patch,

            VH_MIN,

            VH_MAX

        )



        vv_features = compute_glcm_features(

            vv_patch,

            VV_MIN,

            VV_MAX

        )



        results.append([


            lat[r],

            lon[c],


            vv[r, c],

            vh[r, c],



            *vh_features,


            *vv_features


        ])



        count += 1




# ==========================================================
# SAVE
# ==========================================================

print()

print("Samples:", count)

print()




columns = [


'lat',
'lon',

'VV_dB',
'VH_dB',



'VH_contrast',
'VH_homogeneity',
'VH_ASM',
'VH_correlation',
'VH_entropy',



'VV_contrast',
'VV_homogeneity',
'VV_ASM',
'VV_correlation',
'VV_entropy'


]




df = pd.DataFrame(

    results,

    columns=columns

)



print(df.describe())



df.to_parquet(

    OUTPUT,

    index=False

)



print()
print("Saved")
print(OUTPUT)
