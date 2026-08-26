from PIL import Image

image = Image.open("../assets/hacker.png")

# Convert the entire image to grayscale
gray = image.convert("L")

# Save grayscale version
gray.save("../assets/hacker_gray.png")

print("Full image converted to grayscale!")
print("Size:", gray.size)