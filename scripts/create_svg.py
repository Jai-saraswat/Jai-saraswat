from pathlib import Path
import base64

image_path = Path("../assets/hacker.png")
output_path = Path("../assets/hacker.svg")

# Read the original image without modifying it
image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

svg = f"""<svg xmlns="http://www.w3.org/2000/svg"
    width="1254"
    height="1254"
    viewBox="0 0 1254 1254">

    <image
        href="data:image/png;base64,{image_data}"
        x="0"
        y="0"
        width="1254"
        height="1254"
        preserveAspectRatio="xMidYMid meet"
    />

</svg>
"""

output_path.write_text(svg, encoding="utf-8")

print("SVG created successfully!")
print(f"Saved to: {output_path}")