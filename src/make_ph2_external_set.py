"""Construction of the external evaluation set with synthetic hair built from PH2 (Section 3.1.2).

Step 1 (prepare): the 21 hair-free PH2 images listed in ph2_hairfree_ids.txt (official PH2 names,
e.g. IMD006) are read from the original PH2 folder, resized to 512 x 512 pixels and saved with the
short names used in the released set (e.g. IMD06.png); the file order fixes the random sequence.
Step 2 (synthesize): synthetic hair is rendered on each image with 3 densities x 3 colors, giving
189 images with hair, their binary hair masks and a manifest (manifest.csv).

Usage:
    python make_ph2_external_set.py prepare   <PH2_root> <ids.txt> <out_dir>/ph2_original
    python make_ph2_external_set.py synthesize <out_dir>/ph2_original <out_dir> [seed=0]

With seed 0 the script reproduces exactly the released set (pixel-identical outputs).
Requirements: numpy, opencv-python, pandas.
"""
import os, sys, glob, cv2, numpy as np, pandas as pd

# ---------------------------------------------------------------- hair simulator
DENSITY = {'s': (7, 12), 'm': (11, 17), 'l': (15, 23)}                 # number of hairs: [low, high) -> 7-11, 11-16, 15-22
COLOUR = {'d': (25, 20, 18), 'b': (85, 55, 35), 'l': (200, 170, 130)}  # RGB base colour: dark, brown, light


def bezier(p, n=200):
    """Cubic Bezier curve c(t) = (1-t)^3 p0 + 3(1-t)^2 t p1 + 3(1-t) t^2 p2 + t^3 p3 (Equation 1)."""
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]


def render_hairs(size, density, colour, rng):
    """Render the hairs of one image. Returns the alpha map and the premultiplied colour."""
    H = W = size; SS = 4                                                   # 4x supersampling for anti-aliasing
    alpha = np.zeros((H * SS, W * SS), np.float32); col = np.zeros((H * SS, W * SS, 3), np.float32)
    for _ in range(rng.integers(*DENSITY[density])):
        start = rng.uniform(-0.1, 1.1, 2) * size                           # p0: slightly larger area than the image
        ang = rng.uniform(0, 2 * np.pi); L = rng.uniform(0.3, 1.0) * size  # orientation and length (30-100%)
        end = start + L * np.array([np.cos(ang), np.sin(ang)])            # p3
        ctrl = [start + (end - start) * f + rng.normal(0, 0.15 * L, 2) for f in (1 / 3, 2 / 3)]  # p1, p2
        pts = bezier(np.array([start, ctrl[0], ctrl[1], end])) * SS
        w0 = rng.uniform(1.5, 4.0) * SS                                    # root thickness (pixels at 512)
        c = np.clip(np.array(COLOUR[colour]) * rng.uniform(0.8, 1.2) + rng.normal(0, 8, 3), 0, 255)
        op = rng.uniform(0.75, 1.0)                                        # opacity
        seg = np.zeros_like(alpha)
        for k in range(len(pts) - 1):
            w = max(1, int(round(w0 * (1 - 0.6 * k / len(pts)))))         # linear tapering to 40% at the tip
            cv2.line(seg, tuple(np.round(pts[k]).astype(int)), tuple(np.round(pts[k + 1]).astype(int)), 1.0, w, cv2.LINE_AA)
        seg *= op; upd = seg > alpha
        col[upd] = c; alpha = np.maximum(alpha, seg)
    prem = cv2.resize(col * alpha[..., None], (W, H), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_AREA)
    prem = cv2.GaussianBlur(prem, (3, 3), 0.6); alpha = cv2.GaussianBlur(alpha, (3, 3), 0.6)
    return alpha, prem


# ---------------------------------------------------------------- step 1: hair-free images
def prepare(ph2_root, ids_file, out):
    os.makedirs(out, exist_ok=True)
    for i in open(ids_file).read().split():
        hits = glob.glob(os.path.join(ph2_root, '**', f'{i}.bmp'), recursive=True)
        hits = [h for h in hits if 'Dermoscopic_Image' in h] or hits
        if not hits: raise FileNotFoundError(i)
        im = cv2.imread(hits[0]); im = cv2.resize(im, (512, 512))
        short = 'IMD' + str(int(i[3:])).zfill(2)     # file names of the released set (e.g. IMD006 -> IMD06)
        cv2.imwrite(os.path.join(out, f'{short}.png'), im)
    print('prepared', len(os.listdir(out)), 'images')


# ---------------------------------------------------------------- step 2: synthetic hair
def synthesize(src, out, seed=0):
    rng = np.random.default_rng(seed)
    for d in ('with_hair', 'mask'): os.makedirs(os.path.join(out, d), exist_ok=True)
    rows = []
    for f in sorted(os.listdir(src)):
        o = os.path.splitext(f)[0]; x = cv2.cvtColor(cv2.imread(os.path.join(src, f)), cv2.COLOR_BGR2RGB).astype(np.float32)
        for dens in ('s', 'm', 'l'):              # low, medium, high density
            for colr in ('b', 'd', 'l'):          # brown, dark, light
                a, c = render_hairs(512, dens, colr, rng)
                y = c + (1 - a[..., None]) * x    # alpha compositing (premultiplied colour)
                m = cv2.dilate((a > 0.3).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))   # binary hair mask
                stem = f'{o}_{dens}_{colr}'
                cv2.imwrite(os.path.join(out, 'with_hair', stem + '.png'), cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(out, 'mask', stem + '.png'), m)
                rows.append(dict(image=stem, original=o, density=dens, colour=colr, hair_fraction=float((m > 0).mean())))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, 'manifest.csv'), index=False)
    print(len(rows), 'images; mean hair fraction by density:', df.groupby('density').hair_fraction.mean().round(4).to_dict())


if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'synthesize': synthesize(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    else: raise SystemExit(__doc__)
