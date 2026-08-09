# Voxelith brand asset generator. Regenerates every file in this
# directory except logo-ops.json / voxelith-logo.vxlt (those are the
# same mark built *in* Voxelith — see logo-ops.json).
#
#   python assets/branding/generate.py
#
# The mark: a runestone. A solid 7x9 tablet (Voxelith = voxel + -lith)
# with the V inscribed on its broad face as thirteen emissive voxels —
# the letter is *made of* voxels, set into the stone. A letterform needs
# a front face to stay symmetric: cutting the V into the silhouette
# reads as a slot in isometric, and floating glyphs shear apart, so the
# glyph lives on the face; the yaw-30 projection below leaves it a
# gentle 14-degree italic instead of the corner view's 30.
#
# Drawing follows the tool's own rendering conventions (src/view.rs):
# true isometric, and the three visible face tones are pairwise
# distinct — same reason view.rs pins LIGHT's components apart. Top is
# brightest, the broad +z face mid, the +x edge darkest.
#
# Needs Pillow (any recent version; ICO append_images needs >= 9.1).
import math
import os
from PIL import Image, ImageDraw

# Rigid yaw/pitch axonometric, turned toward the face: yaw 30 deg
# (45 is the classic corner view), pitch tan 0.48, y kept unit height.
# Derivation: cb = cos(atan(TAN_PITCH));
#   AX = cos(yaw)/cb   BX = sin(yaw)/cb
#   AY = sin(yaw)*TAN_PITCH   BY = cos(yaw)*TAN_PITCH
# The glyph on the face picks up only atan(AY/AX) = 14 deg of slant —
# a light italic — while the top and side keep the voxel mass visible.
AX, BX = 0.9605, 0.5546
AY, BY = 0.2400, 0.4157

def project(x, y, z):
    """World (x right, y up, z toward lower-left) -> screen (y down)."""
    return (AX * x - BX * z, AY * x + BY * z - y)

def cube_faces(x, y, z):
    """The three camera-facing faces of the unit cube at (x,y,z)."""
    x1, y1, z1 = x + 1, y + 1, z + 1
    top   = [(x, y1, z), (x1, y1, z), (x1, y1, z1), (x, y1, z1)]
    left  = [(x, y, z1), (x1, y, z1), (x1, y1, z1), (x, y1, z1)]
    right = [(x1, y, z), (x1, y, z1), (x1, y1, z1), (x1, y1, z)]
    return top, left, right

def hx(s):
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))

def lerp(c1, c2, f):
    return tuple(round(a + (b - a) * f) for a, b in zip(c1, c2))

def svg_hex(c):
    return '#%02X%02X%02X' % c

# Basalt stone, amber emissive. The amber is the `emissive` material
# flag the tool actually ships — the logo is a feature demo.
PAL = dict(top=hx('A8B3CF'), left=hx('6A7694'), right=hx('3F4760'),
           emis=hx('FFC46B'), glow=hx('FFA43F'))
EMIS_RAMP = {'top': hx('FFDD9E'), 'left': hx('FFBE5C'), 'right': hx('E89A33')}
PLATE, PLATE_RIM = hx('1C2130'), hx('333B52')

SLAB_W, SLAB_H = 7, 9
CUBES = [(x, y, 0) for x in range(SLAB_W) for y in range(SLAB_H)]
# 5x7 V glyph with one stone cell of margin on every side
RUNE_V = ([(1, y) for y in range(4, 8)] + [(5, y) for y in range(4, 8)] +
          [(2, y) for y in (2, 3)] + [(4, y) for y in (2, 3)] + [(3, 1)])
# face-mode per emissive cell: the glyph glows on the broad +z face only
EMISSIVE = {(x, y, 0): 'left' for x, y in RUNE_V}

# The light follows the glyph: every rune cell is its own glow source,
# so the halo is V-shaped rather than one round blob hanging over the
# letter. Tight radius keeps the warm rim on the strokes and leaves the
# outer stone cold.
GLOW_RADIUS = 1.5   # in cube units, per source
GLOW_GAIN = 0.42

# ------------------------------------------------------------------ layout

def fit(size, margin):
    """Scale/offset mapping world -> a size x size canvas, plus the
    screen-space glow centers (one per emissive face)."""
    pts = [project(*c) for cube in CUBES for f in cube_faces(*cube) for c in f]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    s = size * (1 - 2 * margin) / span
    ox = (size - (max(xs) - min(xs)) * s) / 2 - min(xs) * s
    oy = (size - (max(ys) - min(ys)) * s) / 2 - min(ys) * s

    def scr(v):
        p = project(*v)
        return (p[0] * s + ox, p[1] * s + oy)

    centers = []
    for cube, mode in EMISSIVE.items():
        top, left, right = cube_faces(*cube)
        face = {'top': top, 'left': left, 'right': right}.get(mode, left)
        ps = [scr(v) for v in face]
        centers.append((sum(q[0] for q in ps) / 4, sum(q[1] for q in ps) / 4))
    bbox = (min(xs) * s + ox, min(ys) * s + oy, max(xs) * s + ox, max(ys) * s + oy)
    return scr, s, centers, bbox

def lit(fill, centroid, glow_cs, s, pal):
    """Stone catches warm light by proximity to the nearest rune cell."""
    dmin = min(math.dist(centroid, g) for g in glow_cs)
    f = max(0.0, 1 - dmin / (GLOW_RADIUS * s)) ** 1.3 * GLOW_GAIN
    return lerp(fill, pal['glow'], f)

# ------------------------------------------------------------------ PNG mark

SS = 4  # supersample

def render_mark(size, pal=PAL, bloom=True, margin=0.15):
    scr_fit = fit(size * SS, margin)
    scr, s, glow_c, _ = scr_fit
    img = Image.new('RGBA', (size * SS, size * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    emis_repaint = []
    for cube in sorted(CUBES, key=sum):          # painter's order, back to front
        top, left, right = cube_faces(*cube)
        mode = EMISSIVE.get(cube)
        for name, poly in (('top', top), ('left', left), ('right', right)):
            p = [scr(v) for v in poly]
            if mode == 'all' or mode == name:
                d.polygon(p, fill=EMIS_RAMP[name])
                emis_repaint.append((p, EMIS_RAMP[name]))
            else:
                c = (sum(q[0] for q in p) / 4, sum(q[1] for q in p) / 4)
                d.polygon(p, fill=lit(pal[name], c, glow_c, s, pal))
    if bloom and size >= 48:
        # small low-alpha halo per rune cell: their union follows the
        # glyph instead of hanging one round blob over it
        rmax = 1.15 * s
        for gx, gy in glow_c:
            for j in range(20, 0, -1):
                r = rmax * j / 20
                alpha = int(18 * (1 - j / 20) ** 1.6) + 2
                gl = Image.new('RGBA', img.size, (0, 0, 0, 0))
                ImageDraw.Draw(gl).ellipse([gx - r, gy - r * 0.9, gx + r, gy + r * 0.9],
                                           fill=pal['glow'] + (alpha,))
                img = Image.alpha_composite(img, gl)
        d = ImageDraw.Draw(img)
        for p, f in emis_repaint:
            d.polygon(p, fill=f)
    return img.resize((size, size), Image.LANCZOS)

# ------------------------------------------------------------------ icons

def plated_icon(size):
    """Dark rounded plate + mark. The plate downscales with BOX because
    LANCZOS rings on the alpha edge and leaves bright specks at the
    corners; the mark keeps LANCZOS for sharpness. Below 48 px the bloom
    margin is spent on mark area instead and the stone is lifted a step,
    or the icon goes muddy in the taskbar."""
    plate = Image.new('RGBA', (size * SS, size * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(plate)
    m = max(1, round(size * SS * 0.02))
    d.rounded_rectangle([m, m, size * SS - m, size * SS - m],
                        radius=size * SS * 0.22, fill=PLATE + (255,),
                        outline=PLATE_RIM + (255,),
                        width=max(1, round(size * SS * 0.014)))
    plate = plate.resize((size, size), Image.BOX)
    if size >= 48:
        margin, pal = 0.12, PAL
    else:
        lift = 0.14 if size <= 24 else 0.08
        pal = {k: (lerp(v, (255, 255, 255), lift) if k in ('top', 'left', 'right') else v)
               for k, v in PAL.items()}
        margin = 0.03 if size <= 24 else 0.05
    inner = round(size * 0.92)
    mark = render_mark(inner, pal=pal, bloom=(size >= 48), margin=margin)
    out = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    out.alpha_composite(plate)
    out.alpha_composite(mark, ((size - inner) // 2, (size - inner) // 2))
    return out

# ------------------------------------------------------------------ SVG

def mark_svg_body(size, margin=0.15):
    scr, s, glow_cs, bbox = fit(size, margin)
    stone, emis = [], []
    for cube in sorted(CUBES, key=sum):
        top, left, right = cube_faces(*cube)
        mode = EMISSIVE.get(cube)
        for name, poly in (('top', top), ('left', left), ('right', right)):
            p = [scr(v) for v in poly]
            d_attr = ' '.join(f'{q[0]:.2f},{q[1]:.2f}' for q in p)
            hot = (mode == 'all' or mode == name)
            if hot:
                fill = svg_hex(EMIS_RAMP[name])
            else:
                c = (sum(q[0] for q in p) / 4, sum(q[1] for q in p) / 4)
                fill = svg_hex(lit(PAL[name], c, glow_cs, s, PAL))
            # same-color stroke seals antialiasing hairlines between faces
            tag = (f'<polygon points="{d_attr}" fill="{fill}" '
                   f'stroke="{fill}" stroke-width="0.6" stroke-linejoin="round"/>')
            (emis if hot else stone).append(tag)
    # one soft halo per rune cell — the union follows the glyph
    r = 1.15 * s
    defs = (f'<radialGradient id="vxglow" cx="50%" cy="50%" r="50%">'
            f'<stop offset="0" stop-color="{svg_hex(PAL["glow"])}" stop-opacity="0.20"/>'
            f'<stop offset="0.55" stop-color="{svg_hex(PAL["glow"])}" stop-opacity="0.07"/>'
            f'<stop offset="1" stop-color="{svg_hex(PAL["glow"])}" stop-opacity="0"/>'
            f'</radialGradient>')
    halos = '\n'.join(f'<ellipse cx="{gx:.2f}" cy="{gy:.2f}" rx="{r:.2f}" '
                      f'ry="{r*0.9:.2f}" fill="url(#vxglow)"/>' for gx, gy in glow_cs)
    body = '\n'.join(stone) + '\n' + halos + '\n' + '\n'.join(emis)
    return defs, body, bbox

GLYPHS = {  # 5x7 pixel caps (I is 3 wide), matching the voxel language
    'V': ["X...X", "X...X", "X...X", "X...X", ".X.X.", ".X.X.", "..X.."],
    'O': [".XXX.", "X...X", "X...X", "X...X", "X...X", "X...X", ".XXX."],
    'X': ["X...X", "X...X", ".X.X.", "..X..", ".X.X.", "X...X", "X...X"],
    'E': ["XXXXX", "X....", "X....", "XXXX.", "X....", "X....", "XXXXX"],
    'L': ["X....", "X....", "X....", "X....", "X....", "X....", "XXXXX"],
    'I': ["XXX", ".X.", ".X.", ".X.", ".X.", ".X.", "XXX"],
    'T': ["XXXXX", "..X..", "..X..", "..X..", "..X..", "..X..", "..X.."],
    'H': ["X...X", "X...X", "X...X", "XXXXX", "X...X", "X...X", "X...X"],
}

def wordmark_rects(text, u, x0, y0, color):
    """Pixel glyphs as <rect>s, horizontal runs plus vertical runs: the
    union overdraws every interior seam, so no antialiasing hairline can
    survive at any raster scale."""
    out, x = [], x0
    for ch in text:
        rows = GLYPHS[ch]
        w = len(rows[0])
        for ry, row in enumerate(rows):
            rx = 0
            while rx < w:
                if row[rx] == 'X':
                    run = rx
                    while run < w and row[run] == 'X':
                        run += 1
                    out.append(f'<rect x="{x + rx*u:.2f}" y="{y0 + ry*u:.2f}" '
                               f'width="{(run-rx)*u:.2f}" height="{u:.2f}" fill="{color}"/>')
                    rx = run
                else:
                    rx += 1
        for cx in range(w):
            ry = 0
            while ry < len(rows):
                if rows[ry][cx] == 'X':
                    run = ry
                    while run < len(rows) and rows[run][cx] == 'X':
                        run += 1
                    if run - ry > 1:
                        out.append(f'<rect x="{x + cx*u:.2f}" y="{y0 + ry*u:.2f}" '
                                   f'width="{u:.2f}" height="{(run-ry)*u:.2f}" fill="{color}"/>')
                    ry = run
                else:
                    ry += 1
        x += (w + 1.4) * u
    return out, x - 1.4 * u

def write_mark_svg(path, size=512):
    defs, body, _ = mark_svg_body(size)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
                f'width="{size}" height="{size}">\n<defs>{defs}</defs>\n{body}\n</svg>\n')
    print('wrote', path)

def write_banner_svg(path, W=1200, H=300):
    """README hero: the mark and wordmark on the brand's dark ground,
    with the amber glow pooled behind the stone. Committed dark on both
    GitHub themes — a deliberate panel, not a theme-dependent lockup.
    Everything is paths and rects; no fonts to fall back."""
    mark_s = 200
    defs, mark_body, _ = mark_svg_body(mark_s, margin=0.04)
    u = 12.6                                    # wordmark pixel unit
    gap = 46
    rects, xend = wordmark_rects('VOXELITH', u, 0, 0, '#E8ECF4')
    word_w = xend
    total = mark_s + gap + word_w
    mx = (W - total) / 2
    my = (H - mark_s) / 2
    wx = mx + mark_s + gap
    wy = (H - 7 * u) / 2
    word, _ = wordmark_rects('VOXELITH', u, wx, wy, '#E8ECF4')
    gcx, gcy = mx + mark_s / 2, H / 2
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}">\n'
            f'<defs>{defs}'
            f'<radialGradient id="vxamb" cx="50%" cy="50%" r="50%">'
            f'<stop offset="0" stop-color="{svg_hex(PAL["glow"])}" stop-opacity="0.13"/>'
            f'<stop offset="1" stop-color="{svg_hex(PAL["glow"])}" stop-opacity="0"/>'
            f'</radialGradient></defs>\n'
            f'<rect x="1.5" y="1.5" width="{W-3}" height="{H-3}" rx="18" '
            f'fill="#14161C" stroke="#2A3145" stroke-width="2"/>\n'
            f'<ellipse cx="{gcx:.0f}" cy="{gcy:.0f}" rx="330" ry="150" fill="url(#vxamb)"/>\n'
            f'<g transform="translate({mx:.2f},{my:.2f})">\n{mark_body}\n</g>\n'
            + '\n'.join(word) + '\n</svg>\n')
    print('wrote', path)

def write_lockup_svg(path, text_color, size=160):
    defs, body, bbox = mark_svg_body(size, margin=0.10)
    u = size * 7.4 / 100
    y0 = (size - 7 * u) / 2 + u * 0.1
    x0 = bbox[2] + size * 0.16
    rects, xend = wordmark_rects('VOXELITH', u, x0, y0, text_color)
    W = xend + size * 0.06
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {size}" '
                f'width="{W:.0f}" height="{size}">\n<defs>{defs}</defs>\n{body}\n'
                + '\n'.join(rects) + '\n</svg>\n')
    print('wrote', path)

# ------------------------------------------------------------------ main

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = lambda name: os.path.join(here, name)
    write_mark_svg(out('voxelith-mark.svg'))
    write_lockup_svg(out('voxelith-logo-dark.svg'), '#E8ECF4')    # on dark bg
    write_lockup_svg(out('voxelith-logo-light.svg'), '#232838')   # on light bg
    write_banner_svg(out('voxelith-banner.svg'))                  # README hero
    sizes = [256, 128, 64, 48, 32, 24, 16]
    icons = {sz: plated_icon(sz) for sz in sizes}
    icons[64].save(out('icon_64.png'))            # embedded as the window icon
    icons[256].save(out('voxelith.ico'), format='ICO',
                    append_images=[icons[s] for s in sizes[1:]],
                    sizes=[(s, s) for s in sizes])
    print('wrote', out('voxelith.ico'))
    with Image.open(out('voxelith.ico')) as probe:
        got = sorted(probe.info.get('sizes', []))
        assert got == [(s, s) for s in sorted(sizes)], got
    print('ico verified:', got)

if __name__ == '__main__':
    main()
