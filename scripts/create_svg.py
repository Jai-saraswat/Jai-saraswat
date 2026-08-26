from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
from PIL import Image
from rembg import remove


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "assets" / "img.png"
OUTPUT = BASE_DIR / "assets" / "profile.svg"


# ============================================================
# ASCII RESOLUTION
# ============================================================

# Your source is 565 x 691.
# 400 gives a very detailed result without making the SVG
# unnecessarily enormous.
COLS = 400

CHAR_ASPECT = 0.48

FONT_SIZE = 4.2
CHAR_WIDTH = 2.55


# ============================================================
# ASCII CHARACTER RAMP
# ============================================================

# Dark -> light
#
# Dense characters represent shadows.
# Spaces represent bright areas.
#
RAMP = "@%#8&o:;,.  "


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

CLAHE_LIMIT = 2.4

# Higher value gives stronger shadows.
GAMMA = 1.28


# ============================================================
# ANIMATION
# ============================================================

# Each row begins shortly after the previous one.
ROW_DELAY = 0.045

# Time for one row to type.
ROW_DURATION = 0.60


# ============================================================
# COLORS
# ============================================================

BACKGROUND = "#0d1117"
TEXT_COLOR = "#ffffff"


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"\nImage not found:\n{INPUT}\n"
    )


# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

print("Loading image...")

original = Image.open(INPUT).convert("RGB")

rgb = np.array(original)

height, width = rgb.shape[:2]

print(
    f"Original image: {width} x {height}"
)


# ============================================================
# REMOVE BACKGROUND
# ============================================================

print("Removing background...")

rgba = remove(original)

rgba = Image.fromarray(
    np.array(rgba)
).convert("RGBA")

print("Background removed.")


# ============================================================
# PUT SUBJECT ON WHITE BACKGROUND
# ============================================================

white_background = Image.new(
    "RGBA",
    rgba.size,
    (255, 255, 255, 255)
)

white_background.alpha_composite(
    rgba
)

rgb = np.array(
    white_background.convert("RGB")
)


# ============================================================
# GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    rgb,
    cv2.COLOR_RGB2GRAY
)


# ============================================================
# MILD NOISE REDUCTION
# ============================================================

gray = cv2.bilateralFilter(
    gray,
    d=5,
    sigmaColor=25,
    sigmaSpace=25
)


# ============================================================
# GLOBAL LOCAL CONTRAST
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=CLAHE_LIMIT,
    tileGridSize=(8, 8)
)

gray = clahe.apply(gray)


# ============================================================
# GLOBAL CONTRAST STRETCH
# ============================================================

low = np.percentile(
    gray,
    1.5
)

high = np.percentile(
    gray,
    98.5
)

if high > low:

    gray = np.clip(
        (
            gray.astype(np.float32)
            - low
        )
        / (high - low)
        * 255.0,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# FACE / UPPER-BODY DETAIL ENHANCEMENT
# ============================================================
#
# We are NOT cropping.
#
# We are only enhancing this region in-place.
#
# The current photo places the face in the upper-middle
# portion of the image.
#
# These coordinates are normalized so they work with the
# current image dimensions.
# ============================================================

face_x1 = int(width * 0.32)
face_x2 = int(width * 0.82)

face_y1 = int(height * 0.04)
face_y2 = int(height * 0.43)

face_region = gray[
    face_y1:face_y2,
    face_x1:face_x2
]


if face_region.size > 0:

    # Stronger local contrast specifically around
    # hair, glasses, eyes, nose and beard.

    face_clahe = cv2.createCLAHE(
        clipLimit=4.0,
        tileGridSize=(6, 6)
    )

    face_region = face_clahe.apply(
        face_region
    )


    # Sharpen facial edges.

    face_blur = cv2.GaussianBlur(
        face_region,
        (0, 0),
        1.0
    )

    face_region = cv2.addWeighted(
        face_region,
        1.35,
        face_blur,
        -0.35,
        0
    )


    # Put enhanced region back into the full image.

    gray[
        face_y1:face_y2,
        face_x1:face_x2
    ] = face_region


# ============================================================
# GAMMA / SHADOW ENHANCEMENT
# ============================================================

normalized = (
    gray.astype(np.float32)
    / 255.0
)

gray = (
    np.power(
        normalized,
        GAMMA
    )
    * 255.0
).clip(
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# PRESERVE DARK DETAILS
# ============================================================

# Strengthen darker pixels without destroying highlights.

dark_mask = gray < 110

gray[dark_mask] = (
    gray[dark_mask].astype(
        np.float32
    )
    * 0.82
).clip(
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# FINAL LIGHT CONTRAST
# ============================================================

# Slight S-curve style contrast.

normalized = (
    gray.astype(np.float32)
    / 255.0
)

contrast = (
    normalized
    - 0.5
) * 1.12 + 0.5

gray = (
    contrast.clip(
        0,
        1
    )
    * 255
).astype(
    np.uint8
)


# ============================================================
# ASCII DIMENSIONS
# ============================================================

rows = max(
    1,
    int(
        height
        / width
        * COLS
        * CHAR_ASPECT
    )
)

print(
    f"ASCII resolution: {COLS} x {rows}"
)


# ============================================================
# RESIZE FOR ASCII SAMPLING
# ============================================================

small = cv2.resize(
    gray,
    (
        COLS,
        rows
    ),
    interpolation=cv2.INTER_AREA
)


# ============================================================
# PIXELS -> ASCII
# ============================================================

ascii_rows = []

ramp_length = len(RAMP)

for y in range(rows):

    chars = []

    for x in range(COLS):

        brightness = int(
            small[y, x]
        )

        index = int(
            brightness
            / 255.0
            * (ramp_length - 1)
        )

        index = max(
            0,
            min(
                index,
                ramp_length - 1
            )
        )

        chars.append(
            RAMP[index]
        )

    ascii_rows.append(
        "".join(chars)
    )


# ============================================================
# SVG DIMENSIONS
# ============================================================

LINE_HEIGHT = (
    FONT_SIZE * 1.15
)

SVG_WIDTH = (
    COLS * CHAR_WIDTH
)

SVG_HEIGHT = (
    rows * LINE_HEIGHT
    + 20
)


# ============================================================
# SVG HEADER
# ============================================================

svg = f'''<svg
xmlns="http://www.w3.org/2000/svg"
width="{SVG_WIDTH}"
height="{SVG_HEIGHT}"
viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">

<rect
x="0"
y="0"
width="{SVG_WIDTH}"
height="{SVG_HEIGHT}"
fill="{BACKGROUND}"/>

<defs>
'''


# ============================================================
# ROW CLIPPING
# ============================================================

row_ends = []

for i, row in enumerate(
    ascii_rows
):

    y = (
        i
        * LINE_HEIGHT
    )

    start = (
        i
        * ROW_DELAY
    )

    # Remove trailing spaces ONLY for determining
    # the animation endpoint.
    trimmed = row.rstrip()

    if trimmed:
        row_end = (
            len(trimmed)
            * CHAR_WIDTH
        )
    else:
        row_end = CHAR_WIDTH

    row_ends.append(
        row_end
    )

    svg += f'''
<clipPath id="row{i}">

<rect
x="0"
y="{y:.2f}"
width="0"
height="{LINE_HEIGHT:.2f}">

<animate
attributeName="width"
from="0"
to="{row_end:.2f}"
begin="{start:.3f}s"
dur="{ROW_DURATION:.3f}s"
fill="freeze"/>

</rect>

</clipPath>
'''


svg += "</defs>\n"


# ============================================================
# DRAW ASCII ROWS
# ============================================================

for i, row in enumerate(
    ascii_rows
):

    y = (
        (i + 1)
        * LINE_HEIGHT
    )

    safe_row = escape(
        row
    )

    svg += f'''
<text
x="0"
y="{y:.2f}"
font-family="DejaVu Sans Mono, Liberation Mono, monospace"
font-size="{FONT_SIZE}px"
font-weight="400"
fill="{TEXT_COLOR}"
xml:space="preserve"
clip-path="url(#row{i})">{safe_row}</text>
'''


# ============================================================
# MOVING CURSOR
# ============================================================

for i, row_end in enumerate(
    row_ends
):

    y = (
        i
        * LINE_HEIGHT
    )

    start = (
        i
        * ROW_DELAY
    )

    svg += f'''
<rect
x="0"
y="{y + 0.4:.2f}"
width="{FONT_SIZE * 0.55:.2f}"
height="{FONT_SIZE:.2f}"
fill="{TEXT_COLOR}">

<animate
attributeName="x"
from="0"
to="{row_end:.2f}"
begin="{start:.3f}s"
dur="{ROW_DURATION:.3f}s"
fill="freeze"/>

<animate
attributeName="opacity"
values="1;1;0"
keyTimes="0;0.92;1"
begin="{start:.3f}s"
dur="{ROW_DURATION:.3f}s"
fill="freeze"/>

</rect>
'''


# ============================================================
# CLOSE SVG
# ============================================================

svg += """
</svg>
"""


# ============================================================
# SAVE
# ============================================================

OUTPUT.write_text(
    svg,
    encoding="utf-8"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("==========================================")
print("HIGH-DETAIL ASCII PROFILE CREATED")
print("==========================================")
print("Input :", INPUT)
print("Output:", OUTPUT)
print("Grid  :", f"{COLS} x {rows}")
print("==========================================")